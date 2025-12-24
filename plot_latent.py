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

print("[INFO] Transform and config set.")

# データセットの読み込み
train_dir = os.path.join(DATASET_DIR, 'training_set')
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"[INFO] Loaded {len(train_dataset)} images from {train_dir}")

# 学習済みモデルのロード（必要に応じてパスを変更）
vae = VAE(img_channels=3, img_size=IMG_SIZE, latent_dim=LATENT_DIM).to(DEVICE)
vae_weight_path = os.path.join(RESULT_DIR, 'vae.pth')
if os.path.exists(vae_weight_path):
    vae.load_state_dict(torch.load(vae_weight_path, map_location=DEVICE))
    print(f"Loaded weights from {vae_weight_path}")
else:
    print(f"Warning: {vae_weight_path} not found. Using randomly initialized model.")
vae.eval()

print("[INFO] Model ready.")

# 潜在変数の取得
all_mu = []
all_labels = []
with torch.no_grad():
    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        mu, _ = vae.encode(imgs)
        all_mu.append(mu.cpu())
        all_labels.append(labels)

print("[INFO] Latent variables collected.")

all_mu = torch.cat(all_mu, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()

print("[INFO] Latent variables concatenated.")

# 2次元に可視化（t-SNEまたはPCA）
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
mu_2d = pca.fit_transform(all_mu)

print("[INFO] PCA completed.")

plt.figure(figsize=(8, 6))

# クラス名取得
class_names = train_dataset.classes  # 例: ['cats', 'dogs']

# 色ごとにラベルを付けてプロット
for class_idx, class_name in enumerate(class_names):
    idx = all_labels == class_idx
    plt.scatter(mu_2d[idx, 0], mu_2d[idx, 1],
                label=class_name, alpha=0.7)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Latent Space Distribution (PCA)')
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'latent_distribution.png'))
plt.show()

print("[INFO] latent_distribution.png saved.")
