import os
import ujson
import numpy as np
import gc
from concurrent.futures import ProcessPoolExecutor
import torch

# -----------------------------
# 글로벌 설정 (원 코드와 동일)
# -----------------------------
batch_size = 10
train_size = 0.8  # merge original training set and test set, then split it manually.
least_samples = batch_size / (1 - train_size)  # least samples for each client

# 저장 성능 옵션: 압축 사용 여부 (True면 기존과 동일하게 .npz + compressed, False면 빠른 .npy)
USE_COMPRESS = True

def check(config_path, train_path, test_path, num_clients, num_classes, niid=False,
          balance=True, partition=None, alpha=0.1):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
           config['num_classes'] == num_classes and \
           config['non_iid'] == niid and \
           config['balance'] == balance and \
           config['partition'] == partition and \
           config['alpha'] == alpha and \
           config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True
    return False


def _to_numpy_indexable(x):
    """
    dataset_content 이 numpy 배열이면 그대로,
    리스트/튜플이면 인덱싱 가능한 object ndarray로 변환.
    이미지/객체 리스트여도 fancy indexing 가능하도록 처리.
    """
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=object)


def separate_data(data, num_clients, num_classes, niid=False, balance=False,
                  partition=None, class_per_client=2, alpha=0.1):
    """
    속도 최적화 버전:
    - 클래스별 인덱스: argsort + searchsorted + split
    - np.append 제거: 리스트에 모아 마지막에 concatenate
    - IID 분할은 np.array_split으로 즉시 분배
    """
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    # 라벨을 numpy로
    if isinstance(dataset_label, torch.Tensor):
        dataset_label = dataset_label.detach().cpu().numpy()
    else:
        dataset_label = np.asarray(dataset_label)

    # 콘텐츠를 인덱싱 가능한 형태로
    dataset_content = _to_numpy_indexable(dataset_content)

    # 빠른 클래스별 인덱스 구성: O(N)
    # 라벨 기준으로 정렬 → searchsorted로 클래스 경계 찾기 → split
    sorted_idx = np.argsort(dataset_label, kind='stable')
    dataset_label_sorted = dataset_label[sorted_idx]
    split_points = np.searchsorted(dataset_label_sorted, np.arange(1, num_classes))
    idxs_sorted = sorted_idx  # 원본 인덱스 보존
    idx_for_each_class = np.split(idxs_sorted, split_points)  # 길이 num_classes, 각 원소는 인덱스 배열

    dataidx_map = {i: [] for i in range(num_clients)}

    if num_clients == 1:
        # 모든 샘플 인덱스를 client 0 에 통채로 할당
        dataidx_map[0] = np.arange(len(dataset_label))
    else:
        # 원 코드 로직 존중: IID가 아니면 'pat' 강제 & class_per_client 유지
        if not niid:
            partition = 'pat'
            class_per_client = num_classes

        if partition == 'pat':
            # 클래스별로 선택된 클라이언트에게 비율 분배
            class_num_per_client = np.full(shape=(num_clients,), fill_value=class_per_client, dtype=int)
            # 리스트 누적 후 최종 concatenate
            list_bins = [[] for _ in range(num_clients)]

            # selected_clients 수를 고정(과분배 방지)
            # = ceil((num_clients/num_classes)*class_per_client)
            max_clients_per_class = int(np.ceil((num_clients / num_classes) * class_per_client))

            for c in range(num_classes):
                pool = [cid for cid in range(num_clients) if class_num_per_client[cid] > 0]
                if len(pool) == 0:
                    # 모든 클라의 class budget이 0인 경우, 아무나 균등 선택
                    pool = list(range(num_clients))
                selected_clients = pool[:max_clients_per_class]  # 앞에서부터 잘라 고정

                idx_c = idx_for_each_class[c]
                num_all = len(idx_c)
                num_sel = len(selected_clients)
                if num_sel == 0:
                    continue

                num_per = num_all / num_sel
                if balance:
                    # 거의 균등 분할
                    num_samples = [int(num_per)] * (num_sel - 1)
                else:
                    # 최소 샘플(least_samples/num_classes)~num_per 사이 랜덤
                    low = max(num_per / 10, least_samples / max(1, num_classes))
                    # 정수 범위 확보
                    low_i = int(max(1, np.floor(low)))
                    high_i = int(max(low_i + 1, np.ceil(num_per)))
                    if high_i <= low_i:
                        high_i = low_i + 1
                    num_samples = np.random.randint(low_i, high_i, size=(num_sel - 1,)).tolist()
                # 마지막은 잔여
                num_samples.append(int(num_all - sum(num_samples)))
                # 경계 보정(음수 방지)
                num_samples = [max(0, n) for n in num_samples]
                # 길이 차이 보정
                if sum(num_samples) > num_all:
                    num_samples[-1] = max(0, num_all - sum(num_samples[:-1]))

                # 슬라이스 할당
                ptr = 0
                for client, n in zip(selected_clients, num_samples):
                    if n <= 0:
                        continue
                    sl = idx_c[ptr:ptr + n]
                    list_bins[client].append(sl)
                    ptr += n
                    class_num_per_client[client] = max(0, class_num_per_client[client] - 1)

            # 최종 concatenate
            for client in range(num_clients):
                if len(list_bins[client]) == 0:
                    dataidx_map[client] = np.empty((0,), dtype=int)
                else:
                    dataidx_map[client] = np.concatenate(list_bins[client], axis=0)

        elif partition == "dir":
            # 디리클레 분할 (빠르게)
            K = num_classes
            # 클래스별 인덱스는 이미 준비됨: idx_for_each_class
            # 최소 샘플 보장 루프 (너무 빡빡하면 2~3회 시도 후 타협)
            attempts = 0
            while True:
                attempts += 1
                idx_batch = [[] for _ in range(num_clients)]
                for k in range(K):
                    idx_k = idx_for_each_class[k]
                    if len(idx_k) == 0:
                        continue
                    idx_k = np.copy(idx_k)
                    np.random.shuffle(idx_k)
                    props = np.random.dirichlet(np.repeat(alpha, num_clients))
                    props = props / props.sum()
                    cuts = (np.cumsum(props) * len(idx_k)).astype(int)[:-1]
                    split_k = np.split(idx_k, cuts)
                    for j in range(num_clients):
                        if len(split_k[j]) > 0:
                            idx_batch[j].append(split_k[j])

                # 최소 사이즈 계산
                sizes = [sum(len(b) for b in idx_batch[j]) for j in range(num_clients)]
                min_size = min(sizes) if len(sizes) > 0 else 0
                if (min_size >= least_samples) or attempts >= 3:
                    # 타협조건: 3회 이상 시도하면 통과
                    break

            for j in range(num_clients):
                dataidx_map[j] = np.concatenate(idx_batch[j], axis=0) if len(idx_batch[j]) else np.empty((0,), dtype=int)
        else:
            # 기본 IID 균등 분할
            idxs = np.arange(len(dataset_label))
            np.random.shuffle(idxs)
            splits = np.array_split(idxs, num_clients)
            for j in range(num_clients):
                dataidx_map[j] = splits[j]

    # assign data (벡터화 인덱싱; 리스트/이미지도 object ndarray면 fancy indexing 가능)
    for client in range(num_clients):
        idxs = dataidx_map[client]
        Xc = dataset_content[idxs]
        yc = dataset_label[idxs]

        X[client] = Xc
        y[client] = np.squeeze(yc)

        uniq, cnts = np.unique(y[client], return_counts=True) if len(y[client]) > 0 else (np.array([], dtype=int), np.array([], dtype=int))
        statistic[client] = [(int(i), int(c)) for i, c in zip(uniq.tolist(), cnts.tolist())]

    # 로그 출력 (원 포맷 유지)
    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    # 메모리 정리
    del data, dataset_label_sorted, idxs_sorted, idx_for_each_class
    gc.collect()

    return X, y, statistic


def _split_one(xi, yi, train_ratio):
    # sklearn.train_test_split 대체: 빠른 랜덤 분할
    n = len(yi)
    if n == 0:
        return {'x': xi, 'y': yi}, {'x': xi, 'y': yi}
    perm = np.random.permutation(n)
    cut = int(n * train_ratio)
    tr_idx, te_idx = perm[:cut], perm[cut:]
    return {'x': xi[tr_idx], 'y': yi[tr_idx]}, {'x': xi[te_idx], 'y': yi[te_idx]}


def split_data(X, y):
    """
    빠른 분할:
    - 벡터화 랜덤 분할
    - 큰 클라이언트 수면 멀티프로세싱 권장 (여기선 단일 프로세스; 필요 시 주석 해제)
    """
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    # ---- 단일 프로세스 버전 (메모리 안정적, 대부분 충분히 빠름)
    for i in range(len(y)):
        tr, te = _split_one(X[i], y[i], train_size)
        train_data.append(tr)
        test_data.append(te)
        num_samples['train'].append(len(tr['y']))
        num_samples['test'].append(len(te['y']))
    # ---- 멀티프로세싱 옵션 (대규모/CPU 많을 때만 사용 권장)
    # with ProcessPoolExecutor() as ex:
    #     results = list(ex.map(lambda args: _split_one(*args), [(X[i], y[i], train_size) for i in range(len(y))]))
    # for tr, te in results:
    #     train_data.append(tr); test_data.append(te)
    #     num_samples['train'].append(len(tr['y']))
    #     num_samples['test'].append(len(te['y']))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()

    # 원 코드 유지: X, y 해제
    del X, y
    # gc.collect()

    return train_data, test_data


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
              num_classes, statistic, niid=False, balance=True, partition=None, start=0, alpha=0.1):
    """
    빠른 저장:
    - 기본값: 무압축 .npy (매우 빠름)
    - 기존과 완전 동일 동작 필요하면 상단 USE_COMPRESS=True 로 설정
    """
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    print("Saving to disk.\n")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    if True or USE_COMPRESS:
        # 원 코드와 같은 포맷 (느림)
        for idx, train_dict in enumerate(train_data):
            with open(os.path.join(train_path, str(idx + (start if start is not None else 0)) + f'_{alpha}.npz'), 'wb') as f:
                np.savez_compressed(f, data=train_dict)
        for idx, test_dict in enumerate(test_data):
            with open(os.path.join(test_path, str(idx + (start if start is not None else 0)) + f'_{alpha}.npz'), 'wb') as f:
                np.savez_compressed(f, data=test_dict)
    else:
        # 빠른 포맷: .npy (allow_pickle=True)
        for idx, train_dict in enumerate(train_data):
            np.save(os.path.join(train_path, str(idx + (start if start is not None else 0)) + f'_{alpha}_train.npy'),
                    train_dict, allow_pickle=True)
        for idx, test_dict in enumerate(test_data):
            np.save(os.path.join(test_path, str(idx + (start if start is not None else 0)) + f'_{alpha}_test.npy'),
                    test_dict, allow_pickle=True)

    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
