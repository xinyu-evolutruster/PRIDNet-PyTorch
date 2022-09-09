from matplotlib import pyplot as plt  # 展示图片
import numpy as np  # 数值处理
import cv2  # opencv库
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 回归分析
from matplotlib import pyplot as plt
import random
from copy import deepcopy

def read_image(img_path):
    # 读取图片
    img = cv2.imread(img_path)
    # 图片是三通道，则采用matplotlib展示时要先转换通道
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(type(img))
    return img

# 展示图片
def plot_image(image, image_title, is_axis=False):
    plt.imshow(image)
    if not is_axis:
        plt.axis('off')
    plt.title(image_title)
    plt.show()


if __name__ == '__main__':

    img = read_image("./img/A.png")
    h, w = img.shape[0:2]
    print("h = {0}, w = {1}".format(h, w))
    for i in range(h):
        for j in range(w):
            img[i, j, 2] = 255
    
    plot_image(img, "test")
    '''
    one = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    zero = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    g = random.sample(one, 3)
    g = g + random.sample(zero, 7)
    random.shuffle(g)
    w = 80
    l = random.sample([i for i in range(w)], int(0.4 * w))
    print(l)
    '''