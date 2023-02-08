import torch
import torchvision

from model import *

nz = 100
gen_num = 64
gen_search_num = 512
gen_mean = 0
gen_std = 1
gen_path = 'checkpoints/generator_199.pth'
dis_path = 'checkpoints/discriminator_199.pth'
gen_img = 'result.png'


@torch.no_grad()
def generate():
    cuda_id = 0
    device_name = "cuda:{}".format(cuda_id)
    device = torch.device(device_name)

    generator, discriminator = Generator().eval(), Discriminator().eval()
    noises = torch.randn(gen_search_num, nz, 1, 1).normal_(gen_mean, gen_std)
    noises = noises.to(device)

    map_location = lambda storage, loc: storage
    discriminator.load_state_dict(torch.load(dis_path, map_location=map_location))
    generator.load_state_dict(torch.load(gen_path, map_location=map_location))
    discriminator.to(device)
    generator.to(device)

    # 生成图片，并计算图片在判别器的分数
    fake_img = generator(noises)
    scores = discriminator(fake_img).detach()

    # 挑选最好的某几张
    indexs = scores.topk(gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_img.data[ii])
    # 保存图片
    torchvision.utils.save_image(torch.stack(result), gen_img, normalize=True, range=(-1, 1))


if __name__ == "__main__":
    generate()
