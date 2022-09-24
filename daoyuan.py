import cv2
import numpy as np

img = cv2.imread('daoyuan.png')
x, y, z = img.shape
cv2.imshow('daoyuan', img)
cv2.waitKey(0)


# def white_balance(img, pct=0.0025):  # 白平衡
#     out_channels = []
#     cumstops = (x*y*pct, x*y*(1-pct))
#     for channel in cv2.split(img):
#         cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0, 256)))
#         low, high = np.searchsorted(cumhist, cumstops)
#         lut = np.concatenate((
#             np.zeros(low),
#             np.around(np.linspace(0, 255, high-low+1)),
#             255*np.ones(255-high)
#         ))
#         out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
#     img = cv2.merge(out_channels)
#     return img


def white_balance(img, mode=1):
    """⽩平衡处理（默认为1均值、2完美反射、3灰度世界、4基于图像分析的偏⾊检测及颜⾊校正、5动态阈值）"""
    # 读取图像
    b, g, r = cv2.split(img)
    # 均值变为三通道
    h, w, c = img.shape
    b_avg, g_avg, r_avg = cv2.mean(b)[0], cv2.mean(g)[0], cv2.mean(r)[0]
    if mode == 1:
        # 默认均值  ---- 简单的求均值白平衡法
        # 求各个通道所占增益
        k = (b_avg + g_avg + r_avg) / 3
        kb, kg, kr = k / b_avg, k / g_avg, k / r_avg
        b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        output_img = cv2.merge([b, g, r])
    elif mode == 2:
        # 完美反射⽩平衡 ---- 依赖ratio值选取⽽且对亮度最⼤区域不是⽩⾊的图像效果不佳。
        output_img = img.copy()
        sum_ = np.double() + b + g + r
        hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
        Y = 765
        num, key = 0, 0
        ratio = 0.01
        while Y >= 0:
            num += hists[Y]
            if num > h * w * ratio / 100:
                key = Y
                break
            Y = Y - 1
        sumkey = np.where(sum_ >= key)
        sum_b, sum_g, sum_r = np.sum(b[sumkey]), np.sum(g[sumkey]), np.sum(r[sumkey])
        times = len(sumkey[0])
        avg_b, avg_g, avg_r = sum_b / times, sum_g / times, sum_r / times
        maxvalue = float(np.max(output_img))
        output_img[:, :, 0] = output_img[:, :, 0] * maxvalue / int(avg_b)
        output_img[:, :, 1] = output_img[:, :, 1] * maxvalue / int(avg_g)
        output_img[:, :, 2] = output_img[:, :, 2] * maxvalue / int(avg_r)
    elif mode == 3:
        # 灰度世界假设
        # 需要调整的RGB分量的增益
        k = (b_avg + g_avg + r_avg) / 3
        kb, kg, kr = k / b_avg, k / g_avg, k / r_avg
        ba, ga, ra = b * kb, g * kg, r * kr
        output_img = cv2.merge([ba, ga, ra])
    elif mode == 4:
        # 基于图像分析的偏⾊检测及颜⾊校正
        I_b_2, I_r_2 = np.double(b) ** 2, np.double(r) ** 2
        sum_I_b_2, sum_I_r_2 = np.sum(I_b_2), np.sum(I_r_2)
        sum_I_b, sum_I_g, sum_I_r = np.sum(b), np.sum(g), np.sum(r)
        max_I_b, max_I_g, max_I_r = np.max(b), np.max(g), np.max(r)
        max_I_b_2, max_I_r_2 = np.max(I_b_2), np.max(I_r_2)
        [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
        [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])
        b0 = np.uint8(u_b * (np.double(b) ** 2) + v_b * b)
        r0 = np.uint8(u_r * (np.double(r) ** 2) + v_r * r)
        output_img = cv2.merge([b0, g, r0])
    elif mode == 5:
        # 动态阈值算法 ---- ⽩点检测和⽩点调整
        # 只是⽩点检测不是与完美反射算法相同的认为最亮的点为⽩点，⽽是通过另外的规则确定
        def con_num(x):
            if x > 0:
                return 1
            if x < 0:
                return -1
            if x == 0:
                return 0

        yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # YUV空间
        (y, u, v) = cv2.split(yuv_img)
        max_y = np.max(y.flatten())
        sum_u, sum_v = np.sum(u), np.sum(v)
        avl_u, avl_v = sum_u / (h * w), sum_v / (h * w)
        du, dv = np.sum(np.abs(u - avl_u)), np.sum(np.abs(v - avl_v))
        avl_du, avl_dv = du / (h * w), dv / (h * w)
        radio = 0.5  # 如果该值过⼤过⼩，⾊温向两极端发展
        valuekey = np.where((np.abs(u - (avl_u + avl_du * con_num(avl_u))) < radio * avl_du)
                            | (np.abs(v - (avl_v + avl_dv * con_num(avl_v))) < radio * avl_dv))
        num_y, yhistogram = np.zeros((h, w)), np.zeros(256)
        num_y[valuekey] = np.uint8(y[valuekey])
        yhistogram = np.bincount(np.uint8(num_y[valuekey].flatten()), minlength=256)
        ysum = len(valuekey[0])
        Y = 255
        num, key = 0, 0
        while Y >= 0:
            num += yhistogram[Y]
            if num > 0.1 * ysum:  # 取前10%的亮点为计算值，如果该值过⼤易过曝光，该值过⼩调整幅度⼩
                key = Y
                break
            Y = Y - 1
        sumkey = np.where(num_y > key)
        sum_b, sum_g, sum_r = np.sum(b[sumkey]), np.sum(g[sumkey]), np.sum(r[sumkey])
        num_rgb = len(sumkey[0])
        b0 = np.double(b) * int(max_y) / (sum_b / num_rgb)
        g0 = np.double(g) * int(max_y) / (sum_g / num_rgb)
        r0 = np.double(r) * int(max_y) / (sum_r / num_rgb)
        output_img = cv2.merge([b0, g0, r0])
    else:
        raise TypeError('mode should be in [1,2,3,4,5]. Got {}'.format(mode))
    output_img = np.uint8(np.clip(output_img, 0, 255))
    return output_img


def med_blur(img):
    # 高斯滤波
    # gau_33 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    # gau_55 = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])
    # img = cv2.filter2D(img, -1, gau_55/sum(sum(gau_55)))
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    # 中值滤波
    img = cv2.medianBlur(img, 3)

    # # 均值滤波
    # img = cv2.blur(img, (3, 3))
    return img


def deposure(img):
    # 直方图均衡化Histogram equalization algorithm for image
    # 把原图像分为三个通道
    img = img.astype(np.uint8)
    b, g, r = cv2.split(img)
    # 对三个通道都进行均衡化
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 最后合并
    img = cv2.merge((bH, gH, rH))
    # 显示原图和处理过的图像的直方图
    return img


def gamma(img):  # gamma函数处理
    gamma = 0.6
    gamma_table = [np.power(i/255, gamma)*255 for i in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表


def lap_blur(img):
    lap_4 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # 锐化
    img = cv2.filter2D(img, -1, kernel=lap_4)
    return img


def low_pass(img):
    b, g, r = cv2.split(img)
    lst = [b, g, r]
    new_lst = []
    # 设置低通滤波器
    crow, ccol = int(x / 2), int(y / 2)  # 中心位置
    mask = np.zeros((x, y), np.uint8)
    mask[crow - 250:crow + 250, ccol - 400:ccol + 400] = 1
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


def edge_enhan(img):
    img_mean = cv2.blur(img, (3, 3))
    mask = img_mean - img
    img = mask*2+img
    img[img > 255] = 255
    img[img < 0] = 0
    return img


img = white_balance(img)
img = gamma(img)
# img = deposure(img)
# img = low_pass(img)
img = med_blur(img)
img = lap_blur(img)
# img = edge_enhan(img)
cv2.imshow('daoyuan_deal.png', img)
cv2.waitKey(0)
