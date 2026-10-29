import numpy as np
import os
import torch
from collections import defaultdict
from pathlib import Path


def read_data(dataset, idx, is_train=True):
    if is_train:
        data_dir = os.path.join('../dataset', dataset, 'train/')
    else:
        data_dir = os.path.join('../dataset', dataset, 'test/')

    file = data_dir + str(idx) + '.npz'
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data


def read_public_data(path, num_classes):
    """Load an explicit public split without taking samples from private/test data.

    Accepts flcore's data={x,y} npz format, or npz with top-level x/y arrays.
    Labels are optional: FedProtoKD trains on teacher pseudo-labels.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"FedProtoKD requires a separate public split: {path}. "
            "Provide --protokd_public_path /path/to/public.npz with x (and optional y). "
            "To preserve existing splits, prepare a NEW dataset using "
            "dataset/prepare_protokd_public.py --source dataset/<source> "
            "--output dataset/<source>_ProtoKD, then select it with --dataset <source>_ProtoKD."
        )
    with np.load(path, allow_pickle=True) as archive:
        data = archive['data'].item() if 'data' in archive else dict(archive)
    if 'x' not in data or len(data['x']) == 0:
        raise ValueError(f"Public split must contain a nonempty x array: {path}")
    x = torch.as_tensor(data['x'], dtype=torch.float32)
    y = torch.as_tensor(data.get('y', np.zeros(len(x))), dtype=torch.long)
    if y.ndim != 1 or len(x) != len(y):
        raise ValueError("Public x and y must have the same number of samples, with 1D labels")
    if (y < 0).any() or (y >= num_classes).any():
        raise ValueError("Public labels are outside [0, num_classes)")
    return torch.utils.data.TensorDataset(x, y)


def read_client_data(dataset, idx, is_train=True, few_shot=0):
    data = read_data(dataset, idx, is_train)
    if "News" in dataset:
        data_list = process_text(data)
    elif "Shakespeare" in dataset:
        data_list = process_Shakespeare(data)
    else:
        data_list = process_image(data)

    if is_train and few_shot > 0:
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    return data_list

def process_image(data):
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]

