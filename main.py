import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from model import VAE


# 設定
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
RESULT_DIR = os.path.join(BASE_DIR, 'result')
os.makedirs(RESULT_DIR, exist_ok=True)
BATCH_SIZE = 128
IMG_SIZE = 64
LATENT_DIM = 128
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# データセットの前処理
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),  # データ拡張
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
optimizer = optim.Adam(vae.parameters(), lr=2e-3)  # 学習率を上げる


# 学習
vae.train()
loss_history = []
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
    avg_loss = total_loss / len(train_loader.dataset)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# 学習後の重みを保存
torch.save(vae.state_dict(), os.path.join(RESULT_DIR, 'vae.pth'))

# 損失グラフの描画・保存
plt.figure()
plt.plot(range(1, EPOCHS+1), loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VAE Training Loss')
plt.grid()
plt.savefig(os.path.join(RESULT_DIR, 'loss_curve.png'))
plt.close()

# テスト画像の表示
vae.eval()
test_dir = os.path.join(DATASET_DIR, 'test_set')
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

imgs, _ = next(iter(test_loader))
imgs = imgs.to(DEVICE)
with torch.no_grad():
    recon_imgs, _, _ = vae(imgs)

# オリジナル画像と再構成画像を表示・保存
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
plt.savefig(os.path.join(RESULT_DIR, 'reconstruction.png'))
plt.show()

# 画像を個別保存
for i in range(8):
    utils.save_image(imgs[i], os.path.join(RESULT_DIR, f'original_{i+1}.png'))
    utils.save_image(recon_imgs[i], os.path.join(RESULT_DIR, f'reconstructed_{i+1}.png'))
