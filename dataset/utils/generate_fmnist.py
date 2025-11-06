import os
import gc
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from utils.dataset_utils import check, separate_data, split_data, save_file

# --------------------------------------------------
# Gaussian Noise Transform
# --------------------------------------------------
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, -1.0, 1.0)
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


# --------------------------------------------------
# Config & Seed
# --------------------------------------------------
torch.multiprocessing.set_sharing_strategy('file_system')
random.seed(2)
np.random.seed(2)
torch.manual_seed(2)

# --------------------------------------------------
# Main Generator
# --------------------------------------------------
def generate_fmnist_chunked(dir_path, num_clients=100, num_classes=10,
                            chunk_size=5000, alpha_list=[0.1, 0.3, 0.5]):
    os.makedirs(dir_path, exist_ok=True)
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    raw_dir = os.path.join(dir_path, "rawdata")

    if check(config_path, train_path, test_path,
             num_clients, num_classes, True, False, "dir"):
        print("이미 존재하는 데이터셋입니다. 종료합니다.")
        return

    # -----------------------------
    # Transform (+Noise)
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
        AddGaussianNoise(0., 0.05)
    ])

    print("🟢 Loading FashionMNIST...")
    trainset = datasets.FashionMNIST(root=raw_dir, train=True, download=True, transform=transform)
    testset = datasets.FashionMNIST(root=raw_dir, train=False, download=True, transform=transform)

    loaders = [
        DataLoader(trainset, batch_size=256, num_workers=4, shuffle=False),
        DataLoader(testset, batch_size=256, num_workers=4, shuffle=False)
    ]

    tmp_dir = os.path.join(dir_path, "tmp_npz")
    os.makedirs(tmp_dir, exist_ok=True)
    npz_files, chunk_id = [], 0

    imgs_buf, lbls_buf = [], []
    total = 0

    # -----------------------------
    # Chunked save
    # -----------------------------
    for loader in loaders:
        for imgs, labels in loader:
            imgs_buf.append(imgs.numpy())
            lbls_buf.append(labels.numpy())
            total += len(labels)
            if total >= chunk_size:
                imgs_np = np.concatenate(imgs_buf, axis=0)
                lbls_np = np.concatenate(lbls_buf, axis=0)
                npz_path = os.path.join(tmp_dir, f"chunk_{chunk_id}.npz")
                np.savez_compressed(npz_path, imgs=imgs_np, labels=lbls_np)
                npz_files.append(npz_path)
                print(f"  └─ Saved chunk {chunk_id}: {imgs_np.shape[0]} samples")
                imgs_buf, lbls_buf, total = [], [], 0
                chunk_id += 1
                gc.collect()

    # 잔여 데이터 저장
    if imgs_buf:
        imgs_np = np.concatenate(imgs_buf, axis=0)
        lbls_np = np.concatenate(lbls_buf, axis=0)
        npz_path = os.path.join(tmp_dir, f"chunk_{chunk_id}.npz")
        np.savez_compressed(npz_path, imgs=imgs_np, labels=lbls_np)
        npz_files.append(npz_path)
        print(f"  └─ Saved final chunk {chunk_id}: {imgs_np.shape[0]} samples")
        gc.collect()

    # -----------------------------
    # Merge + Dirichlet split
    # -----------------------------
    print("\n[✓] Merging all chunks sequentially...")
    imgs_all, labels_all = [], []
    for npz_path in npz_files:
        data = np.load(npz_path)
        imgs_all.append(data["imgs"])
        labels_all.append(data["labels"])
        data.close()
        gc.collect()

    X = np.concatenate(imgs_all, axis=0)
    y = np.concatenate(labels_all, axis=0)
    del imgs_all, labels_all
    gc.collect()

    num_classes = int(y.max() + 1)
    print(f"[✓] Combined dataset size: {X.shape[0]} samples")

    # α별 분할
    for alpha in alpha_list:
        print(f"\n--- α = {alpha} ---")
        Xp, yp, statistic = separate_data(
            (X, y),
            num_clients=num_clients,
            num_classes=num_classes,
            niid=True, balance=False,
            partition="dir", class_per_client=2, alpha=alpha
        )
        tr, te = split_data(Xp, yp)
        save_file(config_path, train_path, test_path,
                  tr, te, num_clients, num_classes,
                  statistic, True, False, "dir", 0, alpha)
        print(f"  [✓] α={alpha} 저장 완료")
        gc.collect()

    print("\n✅ FashionMNIST 생성 완료 (노이즈 + 청크 분할).")


if __name__ == "__main__":
    generate_fmnist_chunked("fmnist/", num_clients=100, num_classes=10, chunk_size=8000)
