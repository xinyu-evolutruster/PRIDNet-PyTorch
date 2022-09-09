import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from network.pridnet import PridNet
from dataset.prid_dataset import Dataset
from PIL import Image, ImageOps
import numpy as np
import utils.img_util as img_util
import math
import cv2

train_root = './data/cropped/train'
# train_orig_root = './data/cropped/train'

process = transforms.Compose([
    transforms.ToTensor()
])

def train(MAX_EPOCH=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = PridNet().cuda()
    # see network structure
    # Adam optimizer: beta1 = 0.9, beta2 = 0.999, epsilon=1e-8
    optimizer = optim.Adam(net.parameters())
    optimizer.zero_grad() # zero the gradient buffers
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98, last_epoch=-1)
    # loss = L1 loss
    criterion = nn.MSELoss()
    # load train dataset
    train_dataset = Dataset(train_root)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    # orig_dataset = Dataset(train_orig_root)
    # orig_loader = DataLoader(orig_dataset, batch_size=1, shuffle=False)

    # test_img_path = './data/cropped_noise/test/24000.jpg'
    # img_pil = Image.open(test_img_path)
    # img_pil = process(img_pil)
    # img_pil = torch.reshape(img_pil, [1, 3, 256, 256])

    # out_test = net(img_pil)
    # out_test = torch.reshape(out_test,[3, 256, 256]).detach().numpy()
    # print("out_test is: shape={0}, type={1}".format(out_test.shape, type(out_test)))
    # out_test = np.moveaxis(out_test, 0, 2)
    # print("out_test is: shape={0}, type={1}".format(out_test.shape, type(out_test)))
    # img_util.plot_image(out_test, "test")

    for epoch in range(MAX_EPOCH):
        net.train()
        for i, data in enumerate(train_loader):
            # load original images
            # data_iter = iter(orig_loader)
            # img, noise_img, _ = data_iter.next()
            # forward

            img, noise_img, _ = data.to(device)

            # print(img.size())
            # print(noise_img.size())
            # t_img = torch.reshape(img, [3, 256, 256]).numpy()
            # t_img = np.moveaxis(t_img, 0, 2)
            # img_util.plot_image(t_img, "test")
            # t_noise_img = torch.reshape(noise_img, [3, 256, 256]).numpy()
            # t_noise_img = np.moveaxis(t_noise_img, 0, 2)
            # img_util.plot_image(t_noise_img, "test")

            outputs = net(noise_img)
            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, img)
            loss.backward()
            # update weights
            optimizer.step()
            scheduler.step()
            # hint
            print("epoch={0}, iteration={1}, loss={2}".format(epoch, i, loss))
            if i % 100 == 0:
                out_test = torch.reshape(outputs, [3, 256, 256]).detach().numpy()
                out_test = np.moveaxis(out_test, 0, 2)
                img_util.plot_image(out_test, "test")
    return net

def test(img_path):
    # model = torch.load('./model.pth')
    img = img_util.read_image(img_path)
    h = img.shape[0]
    w = img.shape[1]
    print("h = {0}, w = {1}".format(h, w))
    pad = 0
    edge = 0
    if h > w:
        edge = int(math.ceil(h / 16) * 16)
        w_pad = edge - w
        h_pad = edge - h
        img = cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        edge = int(math.ceil(w / 16) * 16)
        w_pad = edge - w
        h_pad = edge - h
        img = cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = process(img)
    img = torch.reshape(img, [1, 3, edge, edge])
    print(img.size())
    net = PridNet()
    out = net(img)
    print(out.size())

if __name__ == '__main__':
    # print(torch.__version__)
    # print(torch.cuda.is_available())
    # trained_net = train()

    # img = img_util.read_image('./img/mona_lisa.pnv')

    # img = preprocess(img)
    # noise_img = preprocess(noise_img)

    test('./img/mona_lisa_noise.png')