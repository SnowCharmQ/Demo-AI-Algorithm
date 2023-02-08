import os.path

import tqdm
import torch
import torchvision
from torch.utils.data import DataLoader

from model import *

gpu = True
cuda_id = 0
device_name = "cuda:{}".format(cuda_id)
device = torch.device(device_name)

env = 'GAN'
data_path = 'data/'
dis_path = None
gen_path = None
save_path = 'img'
lr1 = 2e-4
lr2 = 2e-4
nz = 100
ngf = 64
ndf = 64
image_size = 96
max_epoch = 200
batch_size = 256
d_every = 1
g_every = 5
save_every = 10

# 数据
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.CenterCrop(image_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.ImageFolder(data_path, transform=transforms)
dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True
                        )

# 网络
generator, discriminator = Generator(), Discriminator()
discriminator.to(device)
generator.to(device)

# 定义优化器和损失
optimizer_g = torch.optim.Adam(generator.parameters(), lr1, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr2, betas=(0.5, 0.999))
criterion = nn.BCELoss().to(device)

# 真图片label为1，假图片label为0
# noises为生成网络的输入
true_labels = torch.ones(batch_size).to(device)
fake_labels = torch.zeros(batch_size).to(device)
fix_noises = torch.randn(batch_size, nz, 1, 1).to(device)
noises = torch.randn(batch_size, nz, 1, 1).to(device)

dis_meter = AverageValueMeter()
gen_meter = AverageValueMeter()

epochs = range(max_epoch)
for epoch in tqdm.tqdm(epochs):
    for ii, (img, _) in enumerate(dataloader):
        real_img = img.to(device)

        if ii % d_every == 0:
            # 训练判别器
            optimizer_d.zero_grad()
            # 尽可能的把真图片判别为正确
            output = discriminator(real_img)
            error_d_real = criterion(output, true_labels)
            error_d_real.backward()

            # 尽可能把假图片判别为错误
            noises.data.copy_(torch.randn(batch_size, nz, 1, 1))
            fake_img = generator(noises).detach()  # 根据噪声生成假图
            output = discriminator(fake_img)
            error_d_fake = criterion(output, fake_labels)
            error_d_fake.backward()
            optimizer_d.step()

            error_d = error_d_fake + error_d_real

            dis_meter.add(error_d.item())

        if ii % g_every == 0:
            # 训练生成器
            optimizer_g.zero_grad()
            noises.data.copy_(torch.randn(batch_size, nz, 1, 1))
            fake_img = generator(noises)
            output = discriminator(fake_img)
            error_g = criterion(output, true_labels)
            error_g.backward()
            optimizer_g.step()
            gen_meter.add(error_g.item())

    if not os.path.exists('img'):
        os.mkdir('img')
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if epoch == 0 or (epoch + 1) % save_every == 0:
        fix_fake_imgs = generator(fix_noises)
        # 保存模型、图片
        torchvision.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (save_path, epoch), normalize=True,
                                     range=(-1, 1))
        torch.save(discriminator.state_dict(), 'checkpoints/discriminator_%s.pth' % epoch)
        torch.save(generator.state_dict(), 'checkpoints/generator_%s.pth' % epoch)
        dis_meter.reset()
        gen_meter.reset()
