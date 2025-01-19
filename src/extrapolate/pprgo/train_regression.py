import logging

import numpy as np
import torch
import torch.nn.functional as F

import wandb

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%m-%d %H:%M")


def run_batch(model, xbs, yb, optimizer, train):
    if train:
        model.train()
    else:
        model.eval()

    if train:
        optimizer.zero_grad()

    with torch.set_grad_enabled(train):
        pred = model(*xbs)
        pred = pred.squeeze(-1)

        loss = F.mse_loss(pred, yb.float())

        if train:
            loss.backward()
            optimizer.step()

    return loss.item()


def train(
    model,
    train_set,
    val_set,
    lr,
    weight_decay,
    max_epochs=200,
    batch_size=512,
    batch_mult_val=1,
    eval_step=1,
    early_stop=False,
    patience=50,
    ex=None,
):
    device = next(model.parameters()).device

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(train_set),
            batch_size=batch_size,
            drop_last=False,
        ),
        batch_size=None,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    step = 0
    best_loss = np.inf

    loss_hist = {"train": [], "val": []}
    if ex is not None:
        ex.current_run.info["train"] = {"loss": []}
        ex.current_run.info["val"] = {"loss": []}

    accumulated_loss = 0.0
    nsamples = 0
    best_state = None
    best_epoch = 0

    for epoch in range(max_epochs):
        for xbs, yb in train_loader:
            xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)

            # Run a single batch
            loss_val = run_batch(model, xbs, yb, optimizer, train=True)
            batch_size_current = yb.shape[0]
            accumulated_loss += loss_val * batch_size_current
            nsamples += batch_size_current

            step += 1
            if step % eval_step == 0:
                train_loss = accumulated_loss / nsamples
                loss_hist["train"].append(train_loss)
                if ex is not None:
                    ex.current_run.info["train"]["loss"].append(train_loss)

                if val_set is not None:
                    sample_size = min(len(val_set), batch_mult_val * batch_size)
                    rnd_idx = np.random.choice(
                        len(val_set), size=sample_size, replace=False
                    )

                    xbs_val, yb_val = val_set[rnd_idx]
                    xbs_val, yb_val = [xb.to(device) for xb in xbs_val], yb_val.to(
                        device
                    )

                    # Evaluate in eval mode (no gradient)
                    model.eval()
                    with torch.no_grad():
                        val_pred = model(*xbs_val)
                        val_pred = val_pred.squeeze(-1)

                        val_loss = F.mse_loss(val_pred, yb_val.float()).item()
                    loss_hist["val"].append(val_loss)

                    if ex is not None:
                        ex.current_run.info["val"]["loss"].append(val_loss)

                    logging.info(
                        f"Epoch {epoch}, step {step}: "
                        f"train_loss={train_loss:.5f}, val_loss={val_loss:.5f}"
                    )

                    # Check if this is the best so far
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_epoch = epoch
                        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

                    # Early stopping
                    elif early_stop and epoch >= best_epoch + patience:
                        logging.info(
                            f"Early stopping at epoch {epoch}, best epoch was {best_epoch}"
                        )
                        model.load_state_dict(best_state)
                        return epoch + 1, loss_hist

                else:
                    logging.info(
                        f"Epoch {epoch}, step {step}: train_loss={train_loss:.5f}"
                    )

        accumulated_loss = 0.0
        nsamples = 0

    if val_set is not None and best_state is not None:
        model.load_state_dict(best_state)

    return epoch + 1, loss_hist
