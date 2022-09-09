from matplotlib import pyplot as plt  # 展示图片
import numpy as np  # 数值处理
import cv2  # opencv库
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 回归分析
from matplotlib import pyplot as plt
#from skimage.measure import compare_ssim as ssim
from scipy import spatial
import random
from copy import deepcopy
import math
import torch
from torchvision import transforms

# RGB
rgb_r = 0
rgb_g = 1
rgb_b = 2

# 读取图片
def read_image(img_path):
    # 读取图片
    img = cv2.imread(img_path)
    # 图片是三通道，则采用matplotlib展示时要先转换通道
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

# 展示图片
def plot_image(image, image_title, is_axis=False):
    print(image.shape)
    print(type(image))
    plt.imshow(image)
    if not is_axis:
        plt.axis('off')
    plt.title(image_title)
    plt.show()

# 保存图片
def save_image(filename, image):
    img = np.copy(image)

    # *** 什么叫从给定数组的形状中删除一维条目？
    img = img.squeeze()

    if img.dtype == np.double:
        img = img * np.iinfo(np.uint8).max
        img = img.asstype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

# 线性比例变换法进行归一化
def normalization(image):
    info = np.iinfo(image.dtype)
    return image.astype(np.double) / info.max

def noise_mask_image(img, noise_ratio=[0.8, 0.4, 0.6]):
    """
    根据题目要求生成受损图片
    :param img: cv2 读取图片,而且通道数顺序为 RGB
    :param noise_ratio: 噪声比率，类型是 List,，内容:[r 上的噪声比率,g 上的噪声比率,b 上的噪声比率]
                        默认值分别是 [0.8,0.4,0.6]
    :return: noise_img 受损图片, 图像矩阵值 0-1 之间，数据类型为 np.array,
             数据类型对象 (dtype): np.double, 图像形状:(height,width,channel),通道(channel) 顺序为RGB
    """
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
    r_mask = random.sample(ones, one_num[rgb_r]) + random.sample(zeros, w - one_num[rgb_r])
    g_mask = random.sample(ones, one_num[rgb_g]) + random.sample(zeros, w - one_num[rgb_g])
    b_mask = random.sample(ones, one_num[rgb_b]) + random.sample(zeros, w - one_num[rgb_b])
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

    return noise_img

def get_noise_mask(noise_img):
    """
    获取噪声图像，一般为 np.array
    :param noise_img: 带有噪声的图片
    :return: 噪声图像矩阵
    """
    # 将图片数据矩阵只包含 0和1,如果不能等于 0 则就是 1。
    return np.array(noise_img != 0, dtype='double')

def compute_error(res_img, img):
    """
    计算恢复图像 res_img 与原始图像 img 的 2-范数
    :param res_img:恢复图像
    :param img:原始图像
    :return: 恢复图像 res_img 与原始图像 img 的2-范数
    """
    error = 0.0
    res_img = np.array(res_img)
    img = np.array(img)
    if res_img.shape != img.shape:
        print("shape error res_img.shape and img.shape %s" % (res_img.shape, img.shape))
        return None
    error = np.sqrt(np.sum(np.power(res_img - img, 2)))
    return round(error, 3)

def calc_ssim(img, img_noise):
    """
    计算恢复图像 res_img 与原始图像 img 的 2-范数
    :param res_img:恢复图像 
    :param img:原始图像 
    :return: 恢复图像 res_img 与原始图像 img 的2-范数
    """
    return ssim(img, img_noise, multichannel=True,
                data_range=img_noise.max()-img_noise.min())

def calc_csim(img, img_noise):
    """
    计算图片的 cos 相似度
    :param img: 原始图片， 数据类型为 ndarray, shape 为[长, 宽, 3]
    :param img_noise: 噪声图片或恢复后的图片，
                      数据类型为 ndarray, shape 为[长, 宽, 3]
    :return:
    """
    img = img.reshape(-1)
    img_noise = img.noise.reshape(-1)
    return 1 - spatial.distance.cosine(img, img_noise)

process = transforms.Compose([
    transforms.ToTensor()
])

def restore_image(noise_img, size=4):
    """
    使用 你最擅长的算法模型 进行图像恢复。
    :param noise_img: 一个受损的图像
    :param size: 输入区域半径，长宽是以 size*size 方形区域获取区域, 默认是 4
    :return: res_img 恢复后的图片，图像矩阵值 0-1 之间，数据类型为 np.array,
            数据类型对象 (dtype): np.double, 图像形状:(height,width,channel), 通道(channel) 顺序为RGB
    """
    # 恢复图片初始化，首先 copy 受损图片，然后预测噪声点的坐标后作为返回值。
    res_img = np.copy(noise_img)

    # 获取噪声图像
    noise_mask = get_noise_mask(noise_img)

    # -------------实现图像恢复代码答题区域----------------------------

    net = torch.load('./pridnet_deep.pth', map_location="cpu")
    net = net.float()
    h = noise_img.shape[0]
    w = noise_img.shape[1]
    edge = 0
    if h > w:
        edge = int(math.ceil(h / 16) * 16)
    else:
        edge = int(math.ceil(w / 16) * 16)
    w_pad = edge - w
    h_pad = edge - h
    noise_img = cv2.copyMakeBorder(noise_img, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    noise_img = process(noise_img)
    noise_img = torch.reshape(noise_img, [1, 3, edge, edge])

    out = net(noise_img.float())
    out_t = torch.reshape(out, [3, edge, edge]).cpu().detach().numpy()
    out_t = np.moveaxis(out_t, 0, 2)
    out_t_cropped = out_t[0:h, 0:w]

    # ---------------------------------------------------------------

    print(out_t_cropped.shape)
    print(out_t_cropped)
    return out_t_cropped

def restore_image2(noise_img, size=4):
    """
    使用 区域二元线性回归模型 进行图像恢复。
    :param noise_img: 一个受损的图像
    :param size: 输入区域半径，长宽是以 size*size 方形区域获取区域, 默认是 4
    :return: res_img 恢复后的图片，图像矩阵值 0-1 之间，数据类型为 np.array,
            数据类型对象 (dtype): np.double, 图像形状:(height,width,channel), 通道(channel) 顺序为RGB
    """
    # 恢复图片初始化，首先 copy 受损图片，然后预测噪声点的坐标后作为返回值。
    res_img = np.copy(noise_img)

    # 获取噪声图像
    noise_mask = get_noise_mask(noise_img)

    # -------------实现图像恢复代码答题区域----------------------------
    rows, cols, channel = res_img.shape
    region = 10  # 10 * 10
    row_cnt = rows // region
    col_cnt = cols // region

    for chan in range(channel):
        for rn in range(row_cnt + 1):
            ibase = rn * region
            if rn == row_cnt:
                ibase = rows - region
            for cn in range(col_cnt + 1):
                jbase = cn * region
                if cn == col_cnt:
                     jbase = cols - region
                x_train = []
                y_train = []
                x_test = []
                for i in range(ibase, ibase+region):
                    for j in range(jbase, jbase+region):
                        if noise_mask[i, j, chan] == 0:  # 噪音点
                            x_test.append([i, j])
                            continue
                        x_train.append([i, j])
                        y_train.append([res_img[i, j, chan]])
                if x_train == []:
                    print("x_train is None")
                    continue
                reg = LinearRegression()
                reg.fit(x_train, y_train)
                pred = reg.predict(x_test)
                for i in range(len(x_test)):
                    res_img[x_test[i][0], x_test[i][1], chan] = pred[i][0]
    res_img[res_img > 1.0] = 1.0
    res_img[res_img < 0.0] = 0.0
    # ---------------------------------------------------------------
    return res_img

if __name__ == '__main__':
    img_path = './img/A.png'
    img = read_image(img_path)
    # plot_image(img, 'original image')
    '''
    测试展示并保存图片
    plot_image(img, 'A')
    save_image(filename='./img/A_{}_img.png'.format("save"), image = img)
    '''
    nor_img = normalization(img)
    noise_img = noise_mask_image(nor_img, [0.5, 0.5, 0.5])
    # print(nor_img)
    if noise_img is not None:
        image_title = "noise_mask_image"
        # plot_image(noise_img, image_title)
        print(compute_error(nor_img, noise_img))
        new_noise_img = deepcopy(noise_img)
        restored_img = restore_image(new_noise_img)
        image_title = "restored_image"
        plot_image(restored_img, image_title)
        print(compute_error(nor_img, restored_img))
    else:
        print("返回值是None，请生成受损图片并返回！")