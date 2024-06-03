import json

import numpy as np

mean_cifar100_syn = (0.5321, 0.5066, 0.4586)
std_cifar100_syn = (0.2673, 0.2564, 0.2761)

data1 = np.load("/ceph/ssd/shared/datasets/cifar100_synthetic/cifar100_50m_part1.npz")
data2 = np.load("/ceph/ssd/shared/datasets/cifar100_synthetic/cifar100_50m_part2.npz")
data3 = np.load("/ceph/ssd/shared/datasets/cifar100_synthetic/cifar100_50m_part3.npz")
data4 = np.load("/ceph/ssd/shared/datasets/cifar100_synthetic/cifar100_50m_part4.npz")

images = np.concatenate(
    (data1["image"], data2["image"], data3["image"], data4["image"])
)
labels = np.concatenate(
    (data1["label"], data2["label"], data3["label"], data4["label"])
)

num_samples = len(labels)
np.random.seed(40)

partial_size = 0.02
assert 0 < partial_size < 0.9, "partial_size should be between 0 and 0.9"

subset_indices = {i: [] for i in range(20)}
subset_indices["train"] = []
subset_indices["test"] = []

num_class = 100
for selected_class in range(num_class):
    indices = np.where(labels == selected_class)
    np.random.shuffle(indices)
    indices = indices[0]
    indices = np.array_split(indices, 20)
    for i, subset in enumerate(indices):
        fractional_set = subset[: int(partial_size * len(subset))]
        subset_indices[i].extend(fractional_set.tolist())  # Convert to list
        subset_indices["train"].extend(fractional_set.tolist())  # Convert to list
        print(
            f"{len(fractional_set)} samples out of {len(subset)} samples are selected for class {selected_class} in subset {i}"
        )
        test_set = subset[int(0.98 * len(subset)) :]
        subset_indices["test"].extend(test_set.tolist())  # Convert to list
        print(
            f"{len(test_set)} samples out of {len(subset)} samples are selected for class {selected_class} in test set"
        )


# Convert numpy int64 to Python int for JSON serialization
def convert_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError


# save subset indices in data/ directory as json
with open(
    f"/nfs/homedirs/dhp/unsupervised-data-pruning/data/subset_indices_synthetic_cifar_1M_total_{int(100*partial_size)}_percentage.json",
    "w",
) as f:
    json.dump(subset_indices, f, default=convert_to_native)
