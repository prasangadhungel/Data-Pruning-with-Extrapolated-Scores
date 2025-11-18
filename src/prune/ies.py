def main(args):
    start_time = time.time()
    global_file_name = f'global_{args.dataset}_{args.model}_{args.optimizer}_k{args.k}_{args.removal_criteria}_ma{args.moving_average_rate}'

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    cudnn.benchmark = True
    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.IMAGENET

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, train_loader, test_loader, num_samples = prepare_data(
        cfg.dataset, cfg.training.batch_size
    )

    logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")

    all_best_accuracies = []
    all_saved_ratios = []


    derivative_order = {"zero_derivative": 0, "first_derivative": 1, "second_derivative": 2, "third_derivative": 3}[
        args.removal_criteria]

    for iternum in range(args.num_iterations):

        setup_seed(iternum)

        # Initialize model, optimizer, and scheduler
        model = get_model(args.model, num_classes).to(device)
        optimizer, scheduler, args = get_optimizer_and_scheduler(args.optimizer, model.parameters(), args)

        file_name = f'{args.dataset}_{args.model}_{args.optimizer}_k{args.k}_{args.threshold}_{args.removal_criteria}_ma{args.moving_average_rate}_{iternum}'
        csvfile = file_name + ".csv"
        with open(csvfile, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Len of Training set', 'Avg Training Loss', 'Avg Test Loss', 'TestAcc'])

        # Initialize datasets and dataloaders
        trainset, train_loader, test_loader, num_samples = prepare_data(cfg.dataset, cfg.training.batch_size)
        train_dataset, test_dataset = get_dataset(args.root_dir, transform)
        train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, args.batch_size, args.num_workers)
        full_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, drop_last=True)
        full_train_dataset = train_dataset

        # Loss history for each sample
        sample_loss_history = {idx: [] for idx in range(len(train_loader.dataset))}
        derivative_history = {idx: [] for idx in range(len(train_loader.dataset))}

        best_test_accuracy = 0
        remaining_loader = None

        total_saved_samples = 0
        total_samples = len(full_train_dataset)
        for epoch in range(args.epochs):

            start_epoch_time = time.time()
            model.train()
            running_loss = 0.0
            start_train_time = time.time()
            for inputs, labels, indices in train_loader:
                if inputs.size(0) == 1:
                    continue

                inputs, labels = inputs.to(device), labels.to(device)
                indices = indices.numpy()

                optimizer.zero_grad()

                outputs = model(inputs)
                criterion = nn.CrossEntropyLoss(reduction='none')
                losses = criterion(outputs, labels)

                for i, index in enumerate(indices):
                    sample_loss_history[index].append(losses[i].item())

                loss = losses.mean()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader.dataset)
            end_train_time = time.time()
            print(f'Training time for epoch {epoch + 1}: {end_train_time - start_train_time:.2f} seconds')

            scheduler.step()

            if remaining_loader is not None:
                start_eval_time = time.time()
                with torch.no_grad():
                    model.eval()
                    for inputs, labels, indices in remaining_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        indices = indices.numpy()

                        outputs = model(inputs)
                        criterion = nn.CrossEntropyLoss(reduction='none')
                        losses = criterion(outputs, labels)

                        for i, index in enumerate(indices):
                            sample_loss_history[index].append(losses[i].item())
                end_eval_time = time.time()
                print(f'Evaluation time for epoch {epoch + 1}: {end_eval_time - start_eval_time:.2f} seconds')

            if args.threshold > 0:
                start_calc_time = time.time()
                for idx, losses in sample_loss_history.items():
                    if len(losses) >= derivative_order + 1:
                        latest_derivative = calculate_derivative(losses, derivative_order)
                        derivative_history[idx].append(latest_derivative)
                        if len(derivative_history[idx]) > args.moving_average_rate + args.k:
                            derivative_history[idx] = derivative_history[idx][-(args.moving_average_rate + args.k):]
                end_calc_time = time.time()
                print(f'Derivative calculation time: {end_calc_time - start_calc_time:.2f} seconds')

                start_excluded_samples_time = time.time()
                excluded_samples = []
                for idx, derivatives in derivative_history.items():
                    if len(derivatives) >= args.moving_average_rate + args.k:
                        ma_derivatives = correct_moving_average_new_new(derivatives, args.moving_average_rate)
                        derivative_sum = np.abs(ma_derivatives[-args.k:]).sum()
                        if derivative_sum < threshold:
                            excluded_samples.append(idx)
                end_excluded_samples_time = time.time()
                print(
                    f'Excluded samples calculation time: {end_excluded_samples_time - start_excluded_samples_time:.2f} seconds')

                if excluded_samples:
                        excluded_set = set(excluded_samples)
                        all_indices = set(range(len(full_train_dataset)))
                        new_indices = list(all_indices - excluded_set)
                        train_dataset = CifarDataset(full_train_dataset.data, full_train_dataset.labels, new_indices,
                                                     transform)
                        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers)

                        remaining_indices = list(excluded_set)
                        remaining_dataset = cifar_dataset(full_train_dataset.data, full_train_dataset.labels,
                                                          remaining_indices,
                                                          transform)
                        remaining_loader = DataLoader(remaining_dataset, batch_size=args.batch_size, shuffle=False,
                                                      num_workers=args.num_workers, pin_memory=True)
                        if excluded_samples:
                            total_saved_samples += len(excluded_samples)
            else:
                saved_ratio = 0
                print("Skipping second derivative calculations and sample exclusion (Baseline method)")

            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels, _ in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_test_loss = test_loss / len(test_loader.dataset)
            if correct > best_test_accuracy:
                best_test_accuracy = correct

            with open(csvfile, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [epoch + 1, len(train_loader.dataset), epoch_loss, avg_test_loss, correct])

            end_epoch_time = time.time()
            print(f'=============Finished epoch {epoch + 1}. Time taken: {end_epoch_time - start_epoch_time:.2f} seconds=============')

        print(f'Finished Iternum {iternum}. Best Test Accuracy: {best_test_accuracy:.4f}')
        all_best_accuracies.append(best_test_accuracy / (len(test_dataset) * 0.01))

        saved_ratio = total_saved_samples / (total_samples * args.epochs)
        all_saved_ratios.append(saved_ratio * 100)

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate statistics
    mean_accuracy = np.mean(all_best_accuracies)
    std_accuracy = np.std(all_best_accuracies)
    mean_saved_ratio = np.mean(all_saved_ratios)

    with open(global_file_name + ".csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Threshold', 'Total Run Time', 'Best Accuracy', 'STD', 'Saved Samples Ratio'])
        writer.writerow([
            f"{args.threshold}",
            f"{total_time:.2f}",
            f"{mean_accuracy:.2f}%",
            f"{std_accuracy:.2f}%",
            f"{mean_saved_ratio:.2f}%"
        ])

    print(f"Total Running Time: {total_time:.2f} seconds")
    print(f"Best Accuracy: {mean_accuracy:.2f}%±{std_accuracy:.2f}%")
    print(f"Saved Samples Ratio: {mean_saved_ratio:.2f}%")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)