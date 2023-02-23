import os
import torchvision
from PIL import Image
import numpy as np

from torch.utils.data import Dataset


def trans(img, size):
    # 图片的宽高
    h, w = img.shape[0:2]
    # 需要的尺寸
    n_w = n_h = size
    # 不改变图像的宽高比例
    scale = min(n_h / h, n_w / w)
    h = int(h * scale)
    w = int(w * scale)
    # 缩放图像
    img = Image.fromarray(img)
    img = img.resize((w, h), resample=Image.BICUBIC)
    img = np.array(img)
    # 上下左右分别要扩展的像素数
    top = (n_h - h) // 2
    left = (n_w - w) // 2
    # 生成一个新的填充过的图像，这里用纯黑色进行填充(0,0,0)
    new_img = np.zeros((n_h, n_w, 3), dtype=np.uint8)
    new_img[:, :, :] = (0, 0, 0)
    new_img[top:top + h, left:left + w, :] = img
    return new_img


class Datasets(Dataset):

    def __init__(self, path):
        self.path = path
        # 语义分割需要的图片的图片和标签
        self.name1 = os.listdir(os.path.join(path, "images"))
        self.name2 = os.listdir(os.path.join(path, "1st_manual"))
        self.to_tensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.name1)

    # 简单的正方形转换，把图片和标签转为正方形
    # 图片会置于中央，两边会填充为黑色，不会失真

    def __getitem__(self, index):
        # 拿到的图片和标签
        name1 = self.name1[index]
        name2 = self.name2[index]
        # 图片和标签的路径
        img_path = [os.path.join(self.path, i) for i in ("images", "1st_manual")]
        # 读取原始图片和标签，并转RGB
        img_o = Image.open(os.path.join(img_path[0], name1)).convert('RGB')
        img_l = Image.open(os.path.join(img_path[1], name2)).convert('RGB')
        img_o = np.array(img_o)
        img_l = np.array(img_l)

        # 转成网络需要的正方形
        img_o = trans(img_o, 256)
        img_l = trans(img_l, 256)

        return self.to_tensor(img_o), self.to_tensor(img_l)
