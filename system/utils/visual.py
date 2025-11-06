import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

@torch.no_grad()
def visualize_client_umap(client_model, dataloader, device="cuda", title="Client Feature UMAP"):
    """
    client_model : torch.nn.Module — 학습 완료된 클라이언트 모델
    dataloader   : torch.utils.data.DataLoader — 해당 클라이언트의 데이터셋
    device       : "cuda" or "cpu"
    """
    client_model.eval()
    features, labels = [], []

    print("📤 Extracting features from client model...")
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # base / backbone feature 추출 (예: model.base)
        if hasattr(client_model, "base"):
            reps = client_model.base(x)
        else:
            reps = client_model(x)

        features.append(reps.detach().cpu())
        labels.append(y.detach().cpu())

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    # 표준화
    features = StandardScaler().fit_transform(features)

    print("🎨 Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )
    emb = reducer.fit_transform(features)

    # 데이터프레임 구성
    df = pd.DataFrame({
        "x": emb[:, 0],
        "y": emb[:, 1],
        "label": labels
    })

    # 시각화
    plt.figure(figsize=(9, 8))
    sns.scatterplot(
        data=df,
        x="x", y="y",
        hue="label",
        palette="tab20",
        s=35,
        alpha=0.85,
        linewidth=0
    )
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(title)
    plt.show()

# ------------------------------------------------------------
# 클라이언트 모델 feature → t-SNE 임베딩 → 라벨별 시각화
# ------------------------------------------------------------
@torch.no_grad()
def visualize_client_tsne(client_model, dataloader, device="cuda", title="Client Feature t-SNE"):
    """
    client_model : torch.nn.Module — 학습 완료된 클라이언트 모델
    dataloader   : torch.utils.data.DataLoader — 해당 클라이언트의 데이터셋
    device       : "cuda" or "cpu"
    """
    client_model.eval()
    features, labels = [], []

    print("📤 Extracting features from client model...")
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # base / backbone feature 추출 (예: model.base)
        if hasattr(client_model, "base"):
            reps = client_model.base(x)
        else:
            reps = client_model(x)
        features.append(reps.detach().cpu())
        labels.append(y.detach().cpu())

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    # 표준화
    features = StandardScaler().fit_transform(features)

    print("🎨 Running t-SNE...")
    n_samples = len(features)
    perplexity = min(30, max(5, (n_samples - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1000,
        learning_rate="auto",
        random_state=42
    )
    emb = tsne.fit_transform(features)

    # 데이터프레임 구성
    import pandas as pd
    df = pd.DataFrame({
        "x": emb[:, 0],
        "y": emb[:, 1],
        "label": labels
    })

    # 시각화
    plt.figure(figsize=(9, 8))
    sns.scatterplot(
        data=df,
        x="x", y="y",
        hue="label",
        palette="tab20",
        s=35,
        alpha=0.85,
        linewidth=0
    )
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(title)

# ------------------------------------------------------------
# 예시 사용법
# ------------------------------------------------------------
# client_idx = 3
# client = server.clients[client_idx]
# loader = client.load_test_data()
# visualize_client_tsne(client.model, loader, title=f"Client {client_idx} t-SNE (Label Distribution)")

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd

@torch.no_grad()
def visualize_client_predictions_tsne(client_model, dataloader, device="cuda", title="Client Prediction t-SNE"):
    """
    client_model : torch.nn.Module
        학습 완료된 클라이언트 모델
    dataloader : torch.utils.data.DataLoader
        해당 클라이언트의 데이터셋
    device : str
        "cuda" 또는 "cpu"
    title : str
        플롯 제목
    """
    client_model.eval()
    all_features, all_labels, all_preds = [], [], []

    print("📤 Extracting features and predictions from client...")
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # feature 추출 (base/backbone 기준)
        if hasattr(client_model, "base"):
            reps = client_model.base(x)
        else:
            reps = client_model(x)

        # 예측
        logits = client_model(x)
        preds = torch.argmax(logits, dim=1)

        all_features.append(reps.detach().cpu())
        all_labels.append(y.detach().cpu())
        all_preds.append(preds.detach().cpu())

    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = torch.cat(all_preds).numpy()

    # 표준화
    features = StandardScaler().fit_transform(features)

    print("🎨 Running t-SNE...")
    n_samples = len(features)
    perplexity = min(30, max(5, (n_samples - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1000,
        learning_rate="auto",
        random_state=42
    )
    emb = tsne.fit_transform(features)

    # -------------------------------------------------
    # 두 시각화: (1) Ground Truth / (2) Predictions
    # -------------------------------------------------
    df_true = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "label": labels})
    df_pred = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "pred": preds})

    plt.figure(figsize=(16, 7))

    # Ground Truth
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_true, x="x", y="y", hue="label",
                    palette="tab20", s=35, alpha=0.8, linewidth=0)
    plt.title("Ground Truth Labels", fontsize=15)
    plt.xlabel("t-SNE Dim 1"); plt.ylabel("t-SNE Dim 2")
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Predictions
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df_pred, x="x", y="y", hue="pred",
                    palette="tab20", s=35, alpha=0.8, linewidth=0)
    plt.xlabel("t-SNE Dim 1"); plt.ylabel("t-SNE Dim 2")
    plt.legend(title="Pred", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.suptitle(title, fontsize=17, y=1.02)
    plt.tight_layout()
    plt.savefig(title)

# -------------------------------------------------
# 사용 예시
# -------------------------------------------------
# client_idx = 2
# client = server.clients[client_idx]
# loader = client.load_test_data()
# visualize_client_predictions_tsne(client.model, loader, title=f"Client {client_idx} t-SNE Mapping")
