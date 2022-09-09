import sys
sys.path.append('..')

from torch.utils import data
from torchvision import transforms
import utils.img_util as img_util
from PIL import Image, ImageOps
import numpy as np
from copy import deepcopy
import random
import os

rgb_r = 0
rgb_g = 1
rgb_b = 2

preprocess = transforms.Compose([
    transforms.ToTensor()
])

def default_loader(path):
    img_pil = Image.open(path)
    img_tensor = preprocess(img_pil)
    return img_tensor

def add_noise(f, mean=0, var=0.001):
    img = np.array(f/255, dtype=float)
    noise = np.random.normal(mean, var**0.5, img.shape)
    out = img + noise
    if out.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out

def noise_mask_image(img, noise_ratio=[0.35, 0.35, 0.35]):
    # 受损图片初始化
    noise_img = None

    # -------------实现受损图像答题区域-----------------

    noise_img = deepcopy(img)
    h, w = img.shape[0:2]
    ones = [1] * w
    zeros = [0] * w
    one_num = [
        int(w * noise_ratio[rgb_r]),
        int(w * noise_ratio[rgb_g]),
        int(w * noise_ratio[rgb_b])
    ]
    r_mask = random.sample(ones, one_num[rgb_r]) + random.sample(zeros, w-one_num[rgb_r])
    g_mask = random.sample(ones, one_num[rgb_g]) + random.sample(zeros, w-one_num[rgb_g])
    b_mask = random.sample(ones, one_num[rgb_b]) + random.sample(zeros, w-one_num[rgb_b])
    for row in range(h):
        random.shuffle(r_mask)
        random.shuffle(g_mask)
        random.shuffle(b_mask)
        for col in range(w):
            noise_img[row, col, rgb_r] = r_mask[col]
            noise_img[row, col, rgb_g] = g_mask[col]
            noise_img[row, col, rgb_b] = b_mask[col]
    # -----------------------------------------------

    noise_img = get_noise_mask(noise_img)
    noise_img = img * noise_img
    noise_img = np.uint8(noise_img)

    return noise_img

def get_noise_mask(noise_img):
    # 将图片数据矩阵只包含 0和1,如果不能等于 0 则就是 1。
    return np.array(noise_img != 0, dtype='float')

class Dataset(data.Dataset):
    def __init__(self, data_root):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.img_lst = []
        for img in os.listdir(self.data_root):
            img_path = os.path.join(self.data_root, img)
            self.img_lst.append(img_path)

    def __getitem__(self, idx):
        img = img_util.read_image(self.img_lst[idx])
        
        # noise_img = add_noise(img, var=0.008)
        # noise_img = add_noise(img, var=0.008)
        noise_img = noise_mask_image(img)

        img = preprocess(img)
        noise_img = preprocess(noise_img)

        return img, noise_img, 0

    def __len__(self):
        size = len(self.img_lst)
        return size

root = '../data/cropped_noise/train'
if __name__ == '__main__':
    train_dataset = Dataset(data_root=root)
    for img in enumerate(train_dataset):
    #    img_util.plot_image(img[1], "test")
        print(img[1].size())
