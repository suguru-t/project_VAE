import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import VAE

# 設定
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
RESULT_DIR = os.path.join(BASE_DIR, 'result')
BATCH_SIZE = 128
IMG_SIZE = 64
LATENT_DIM = 128  # main.pyと合わせる
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# データセットの前処理
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# データセットの読み込み
train_dir = os.path.join(DATASET_DIR, 'training_set')
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 学習済みモデルのロード（必要に応じてパスを変更）
vae = VAE(img_channels=3, img_size=IMG_SIZE, latent_dim=LATENT_DIM).to(DEVICE)
# ここで重みをロードしたい場合は以下を有効化
# vae.load_state_dict(torch.load('vae.pth', map_location=DEVICE))
vae.eval()

# 潜在変数の取得
all_mu = []
all_labels = []
with torch.no_grad():
    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        mu, _ = vae.encode(imgs)
        all_mu.append(mu.cpu())
        all_labels.append(labels)

all_mu = torch.cat(all_mu, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()

# 2次元に可視化（t-SNEまたはPCA）
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
mu_2d = pca.fit_transform(all_mu)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(mu_2d[:, 0], mu_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label='Class')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Latent Space Distribution (PCA)')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'latent_distribution.png'))
plt.show()
