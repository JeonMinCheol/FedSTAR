import os, gc, numpy as np, random, torch, torchvision
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import DataLoader
from utils.dataset_utils import check, separate_data, split_data, save_file
import torchvision.transforms as transforms

# --------------------------------------------------
# Gaussian Noise Transform
# --------------------------------------------------
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean; self.std = std
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, -1.0, 1.0)
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

# reproducibility
torch.multiprocessing.set_sharing_strategy('file_system')
random.seed(1); np.random.seed(1)

domain_names = ['amazon/', 'dslr/', 'webcam/']
num_classes = 31


# α 분할 워커 (memmap 직접 접근)
def _alpha_partition_worker(args):
    imgs_mm_path, imgs_shape, lbls_mm_path, lbls_shape, num_clients, num_classes, \
        config_path, train_path, test_path, alpha = args
    imgs = np.memmap(imgs_mm_path, mode='r', dtype=np.float32, shape=imgs_shape)
    lbls = np.memmap(lbls_mm_path, mode='r', dtype=np.int64,   shape=lbls_shape)

    X, y, statistic = separate_data(
        (imgs, lbls),
        num_clients=num_clients, num_classes=num_classes,
        niid=True, balance=False, partition="dir", class_per_client=2, alpha=alpha
    )
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path,
              train_data, test_data,
              num_clients, num_classes, statistic,
              niid=True, balance=False, partition="dir", start=0, alpha=alpha)
    return alpha


def generate_office31_mixed(dir_path, num_clients, num_classes,
                             niid=True, balance=False, partition="dir",
                             chunk_size=5000):
    os.makedirs(dir_path, exist_ok=True)
    train_path, test_path = os.path.join(dir_path, "train"), os.path.join(dir_path, "test")
    os.makedirs(train_path, exist_ok=True); os.makedirs(test_path, exist_ok=True)
    config_path = os.path.join(dir_path, "config.json")

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        print("이미 데이터셋이 존재합니다. 건너뜁니다."); return

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        AddGaussianNoise(0.0, 0.05)
    ])

    tmp_dir = os.path.join(dir_path, "tmp_npz"); os.makedirs(tmp_dir, exist_ok=True)
    npz_files = []
    print("\n===== Office-31 Domain Loading (fast) =====")

    # 각 도메인별 로딩 + 청크 저장
    for dom in domain_names:
        domain_dir = os.path.join(dir_path, dom)
        if not os.path.isdir(domain_dir):
            print(f"[Skip] {domain_dir} 없음"); continue

        dataset = torchvision.datasets.ImageFolder(root=domain_dir, transform=transform)
        print(f"Loaded {dom}: {len(dataset)} images")

        loader = DataLoader(dataset, batch_size=128, shuffle=False,
                            num_workers=8, pin_memory=True,
                            prefetch_factor=4, persistent_workers=True)

        imgs_t, lbls_t, chunk_idx = [], [], 0
        for batch_imgs, batch_lbls in tqdm(loader, desc=f"Processing {dom}"):
            imgs_t.append(batch_imgs.cpu()); lbls_t.append(batch_lbls.cpu())
            if sum(x.shape[0] for x in imgs_t) >= chunk_size:
                cat_i, cat_l = torch.cat(imgs_t), torch.cat(lbls_t)
                npz_path = os.path.join(tmp_dir, f"{dom[:-1]}_{chunk_idx}.npz")
                np.savez_compressed(npz_path, imgs=cat_i.numpy(), labels=cat_l.numpy())
                npz_files.append(npz_path)
                print(f"  └─ Saved chunk {chunk_idx}: {cat_i.shape[0]} samples")
                imgs_t.clear(); lbls_t.clear(); gc.collect(); chunk_idx += 1
        if imgs_t:
            cat_i, cat_l = torch.cat(imgs_t), torch.cat(lbls_t)
            npz_path = os.path.join(tmp_dir, f"{dom[:-1]}_{chunk_idx}.npz")
            np.savez_compressed(npz_path, imgs=cat_i.numpy(), labels=cat_l.numpy())
            npz_files.append(npz_path)
            print(f"  └─ Saved final chunk {chunk_idx}: {cat_i.shape[0]} samples")
            imgs_t.clear(); lbls_t.clear(); gc.collect()

    # ----------------------------------------------
    # memmap 병합
    # ----------------------------------------------
    print("\n[✓] 모든 NPZ 병합 (memmap)...")
    total, sample_shape = 0, None
    for f in npz_files:
        with np.load(f) as d:
            total += d["imgs"].shape[0]
            if sample_shape is None:
                sample_shape = d["imgs"].shape[1:]
    C, H, W = sample_shape
    imgs_mm_path, lbls_mm_path = os.path.join(tmp_dir,"all_imgs.mm"), os.path.join(tmp_dir,"all_lbls.mm")

    imgs_mm = np.memmap(imgs_mm_path, mode='w+', dtype=np.float32, shape=(total, C, H, W))
    lbls_mm = np.memmap(lbls_mm_path, mode='w+', dtype=np.int64,   shape=(total,))
    off = 0
    for f in npz_files:
        with np.load(f) as d:
            x, y = d["imgs"], d["labels"]; n = x.shape[0]
            imgs_mm[off:off+n], lbls_mm[off:off+n] = x, y; off += n
    del imgs_mm, lbls_mm; gc.collect()
    print(f"[✓] Combined dataset memmap: {total} samples")

    # ----------------------------------------------
    # α 병렬 Dirichlet 분할 및 저장
    # ----------------------------------------------
    num_classes_detected = int(np.memmap(lbls_mm_path, mode='r', dtype=np.int64, shape=(total,)).max()) + 1
    print(f"Detected {num_classes_detected} classes")

    alphas = [0.1, 0.3, 0.5]
    tasks = [
        (imgs_mm_path, (total, C, H, W),
         lbls_mm_path, (total,),
         num_clients, num_classes_detected,
         config_path, train_path, test_path, alpha)
        for alpha in alphas
    ]
    print("\n[✓] Dirichlet split 병렬 처리...")
    with ProcessPoolExecutor(max_workers=len(alphas)) as ex:
        futures = [ex.submit(_alpha_partition_worker, t) for t in tasks]
        for f in futures:
            print(f"[✓] α={f.result()} 저장 완료.")

    print("\n✅ Office-31 전체 도메인 통합 후 클라이언트 분할 완료 (포맷 동일).")


if __name__ == "__main__":
    generate_office31_mixed(
        dir_path="Office-31/",
        num_clients=100,
        num_classes=num_classes,
        niid=True,
        balance=False,
        partition="dir",
        chunk_size=5000
    )
