import cv2
import numpy as np

img = cv2.imread('fushishan.png')
x, y, z = img.shape
cv2.imshow('fuji', img)
cv2.waitKey(0)


def white_balance(img):
    balancer = cv2.xphoto.createLearningBasedWB()
    img = balancer.balanceWhite(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    h = h*1.15
    s = s*0.8
    h = h.astype(np.uint8)
    s = s.astype(np.uint8)
    v = v.astype(np.uint8)
    h[h > 180] = 180
    s[s > 255] = 255
    v[v > 255] = 255
    img = cv2.merge([h, s, v])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # H = np.zeros((x, y), np.uint8)
    # S = np.zeros((x, y), np.uint8)
    # V = np.zeros((x, y), np.uint8)
    # r, g, b = cv2.split(img)
    # r, g, b = r / 255.0, g / 255.0, b / 255.0
    # for i in range(0, x):
    #     for j in range(0, y):
    #         mx = max((b[i, j], g[i, j], r[i, j]))
    #         mn = min((b[i, j], g[i, j], r[i, j]))
    #         dt = mx - mn
    #
    #         if mx == mn:
    #             H[i, j] = 0
    #         elif mx == r[i, j]:
    #             if g[i, j] >= b[i, j]:
    #                 H[i, j] = (60 * ((g[i, j]) - b[i, j]) / dt)
    #             else:
    #                 H[i, j] = (60 * ((g[i, j]) - b[i, j]) / dt) + 360
    #         elif mx == g[i, j]:
    #             H[i, j] = 60 * ((b[i, j]) - r[i, j]) / dt + 120
    #         elif mx == b[i, j]:
    #             H[i, j] = 60 * ((r[i, j]) - g[i, j]) / dt + 240
    #         H[i, j] = int(H[i, j] / 2)
    #         # S
    #         if mx == 0:
    #             S[i, j] = 0
    #         else:
    #             S[i, j] = int(dt / mx * 255)
    #         # V
    #         V[i, j] = int(mx * 255)
    # img = cv2.merge([H, S, V])
    # img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


def gamma(img):  # gamma函数处理
    gamma = 0.6
    gamma_table = [np.power(i/255, gamma)*255 for i in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表


def low_pass(img):
    b, g, r = cv2.split(img)
    lst = [b, g, r]
    new_lst = []
    # 设置低通滤波器
    crow, ccol = int(x / 2), int(y / 2)  # 中心位置
    mask = np.zeros((x, y), np.uint8)
    mask[crow - 200:crow + 200, ccol - 350:ccol + 350] = 1
    for i in lst:
        # 傅里叶变换
        dft = np.fft.fft2(i)
        fshift = np.fft.fftshift(dft)
        # 掩膜图像和频谱图像乘积
        f = fshift * mask
        # 傅里叶逆变换
        ishift = np.fft.ifftshift(f)
        iimg = np.fft.ifft2(ishift)
        iimg = np.abs(iimg)  # 进行处理，调整大小，获得最终结果, 出来的是复数，无法显示
        ii = (iimg - np.amin(iimg)) / (np.amax(iimg) - np.amin(iimg))  # 调整大小范围便于显示
        new_lst.append(ii)
    img = cv2.merge(new_lst)
    return img


img = gamma(img)
img = white_balance(img)
img = low_pass(img)
cv2.imshow('fuji_deal.png', img)
cv2.waitKey(0)
