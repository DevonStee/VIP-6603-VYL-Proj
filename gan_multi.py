# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os

# %%
# Set random seed

manualSeed = 999

torch.manual_seed(manualSeed)

# %%
dataroot = "./data"
image_size = 64
batch_size = 128
num_epochs = 100
lr = 0.0002
beta1 = 0.5

# %%
from torch.utils.data import Subset
selected_classes = [1, 5, 7]  # automobile, dog, horse

dataset = dset.CIFAR10(root=dataroot, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(image_size),
                           transforms.CenterCrop(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ]))


filtered_indices = [i for i, label in enumerate(dataset.targets) if label in selected_classes]

# 构建只包含这三个类别的子集数据集
small_dataset = Subset(dataset, filtered_indices)

# 构建 DataLoader
dataloader = torch.utils.data.DataLoader(small_dataset, batch_size=batch_size, shuffle=True)

nz = 100  # Size of the random noise vector
ngf = 64  # Depth of the generator feature maps
ndf = 64  # Depth of the discriminator feature maps
nc = 3  # Number of image channels (RGB)


# %%
# 定义生成器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入：nz x 1 x 1 -> 输出：ngf*8 x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 状态：ngf*8 x 4 x 4 -> ngf*4 x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 状态：ngf*4 x 8 x 8 -> ngf*2 x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 状态：ngf*2 x 16 x 16 -> ngf x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 状态：ngf x 32 x 32 -> nc x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # 输出范围[-1, 1]
        )
    def forward(self, input):
        return self.main(input)



# %%
# 定义判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入：nc x 64 x 64 -> ndf x 32 x 32
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态：ndf x 32 x 32 -> ndf*2 x 16 x 16
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态：ndf*2 x 16 x 16 -> ndf*4 x 8 x 8
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态：ndf*4 x 8 x 8 -> ndf*8 x 4 x 4
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态：ndf*8 x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # 输出真实度概率
        )
    def forward(self, input):
        return self.main(input)

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current using:", device)

# 创建模型并应用权重初始化
netG = Generator().to(device)
netD = Discriminator().to(device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

# 如果有多个 GPU，则用 DataParallel 包裹
if torch.cuda.device_count() > 1:
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

# %%
import time
os.makedirs("output_m", exist_ok=True)
real_label = 0.9  # 使用 label smoothing: 真实标签取 0.9
fake_label = 0.0

for epoch in range(num_epochs):
    epoch_start = time.time()
    last_real = None
    last_fake = None

    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) 更新判别器
        ############################
        netD.zero_grad()
        real, _ = data
        real = real.to(device)
        b_size = real.size(0)
        label = torch.full((b_size,), real_label, device=device)  # 真实样本标签：0.9
        
        output = netD(real).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        
        ############################
        # (2) 更新生成器
        ############################
        netG.zero_grad()
        label.fill_(real_label)  # 生成器希望判别器认为生成图像是真实的（即 0.9）
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        if i % 50 == 0:
            print(f"[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}] "
                  f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                  f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")
        
        # 保存最后一个 batch 的真实与生成图像用于对比展示
        last_real = real
        last_fake = fake.detach()

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f} seconds.")
    
    # 拼接最后一个 batch 的真实与生成图像（按宽度拼接）
    if last_real is not None and last_fake is not None:
        comparison = torch.cat((last_real, last_fake), dim=3)
        vutils.save_image(comparison, f"output_m/comparison_epoch_{epoch:03d}.png", normalize=True)
    
    # 保存生成图像
    vutils.save_image(last_fake, f"output_m/fake_samples_epoch_{epoch:03d}.png", normalize=True)


