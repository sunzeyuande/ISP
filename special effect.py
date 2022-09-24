import cv2
import numpy as np

img = cv2.imread('zhoukang.jpg')
lst = []


# def spin(img, degree):
#     x, y, _ = img.shape
#     matrix = cv2.getRotationMatrix2D((int(x/2), int(y/2)), degree, 1)
#     img_spin = cv2.warpAffine(img, matrix, (img.shape[0], img.shape[1]))
#     return img_spin


def rotate_bound(image, degree):
    # grab the dimensions of the image and then determine the
    # center 获取图像的尺寸，然后确定中心
    h, w = image.shape[:2]
    cX, cY = w//2, h//2
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # 获取旋转矩阵(应用的负的角度顺时针旋转)，然后抓取正弦和余弦
    # (即矩阵的旋转分量)
    M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image（计算图像的新边界尺寸）
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation(调整旋转矩阵以考虑到平移
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    image = cv2.warpAffine(image, M, (nW, nH))
    # perform the actual rotation and return the image执行实际的旋转并返回图像
    # M表示旋转矩阵，cv2.warpAffine(image, M, (nW, nH),borderValue=(255,255,255))表示旋转之后的图片,
    # borderValue=(255,255,255)表示旋转之后填充的颜色（255表示白色，不写默认是0是黑色）
    lst.append(image)
    return


for i in range(1, 361):
    rotate_bound(img, i)

video = cv2.VideoWriter('zhoukang.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 60, (1920, 1080))

for i in range(1, 361):
    image = cv2.resize(lst[i-1], (int(i*3*16/9), i*3))
    top = 0
    bottom = 1080-image.shape[0]
    left = 1920-image.shape[1]
    right = 0
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    video.write(image)
video.release()
cv2.destroyAllWindows()
