import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from model import VAE

# 設定
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
BATCH_SIZE = 64
IMG_SIZE = 64
LATENT_DIM = 128
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# データセットの前処理
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# データセットの読み込み
train_dir = os.path.join(DATASET_DIR, 'training_set')
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# モデル・最適化
vae = VAE(img_channels=3, img_size=IMG_SIZE, latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# 学習
vae.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(DEVICE)
        optimizer.zero_grad()
        recon_imgs, mu, logvar = vae(imgs)
        loss = loss_function(recon_imgs, imgs, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader.dataset):.4f}")

# テスト画像の表示
vae.eval()
test_dir = os.path.join(DATASET_DIR, 'test_set')
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

imgs, _ = next(iter(test_loader))
imgs = imgs.to(DEVICE)
with torch.no_grad():
    recon_imgs, _, _ = vae(imgs)

# オリジナル画像と再構成画像を表示
imgs = imgs.cpu()
recon_imgs = recon_imgs.cpu()
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    axes[0, i].imshow(imgs[i].permute(1, 2, 0))
    axes[0, i].axis('off')
    axes[1, i].imshow(recon_imgs[i].permute(1, 2, 0))
    axes[1, i].axis('off')
axes[0, 0].set_ylabel('Original')
axes[1, 0].set_ylabel('Reconstructed')
plt.tight_layout()
plt.show()
