import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

nx = 9
ny = 7
# 获取棋盘格
file_paths = glob.glob("./chess/chess*.JPG")


# 绘制对比图
def plot_contrast_image(origin_img, converted_img, origin_img_title="origin_img", converted_img_title="converted_img"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 20))
    ax1.set_title = origin_img_title
    ax1.imshow(origin_img)
    ax2.set_title = converted_img_title
    ax2.imshow(converted_img)
    plt.show()


# 相机矫正使用opencv封装好的api
# 目的:得到内参、外参、畸变系数
def cal_calibrate_params(file_paths):
    # 存储角点数据的坐标
    object_points = []  # 角点在真实三维空间的位置
    image_points = []  # 角点在图像空间中的位置
    # 生成角点在真实世界中的位置
    objp = np.zeros(((nx-1) * (ny-1), 3), np.float32)
    # 以棋盘格作为坐标，每相邻的黑白棋的相差1
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    # 角点检测
    for file_path in file_paths:
        img = cv2.imread(file_path)
        # 将图像灰度化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 角点检测
        rect, coners = cv2.findChessboardCorners(gray, (nx-1, ny-1), cv2.CALIB_CB_FILTER_QUADS)

        # 角点检测结果的绘制
        imgcopy = img.copy()
        cv2.drawChessboardCorners(imgcopy, (nx-1, ny-1), coners, rect)
        plot_contrast_image(img, imgcopy)

        # 若检测到角点，则进行保存 即得到了真实坐标和图像坐标
        if rect:
            object_points.append(objp)
            image_points.append(coners)
    # 相机矫正
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


# 图像去畸变：利用相机校正的内参，畸变系数
def img_undistort(img, mtx, dist):
    dis = cv2.undistort(img, mtx, dist, None, mtx)
    return dis


ret, mtx, dist, rvecs, tvecs = cal_calibrate_params(file_paths)
img1 = cv2.imread('distortion1.bmp')
img2 = cv2.imread('distortion2.bmp')
img3 = cv2.imread('distortion3.bmp')

undistort_img1 = img_undistort(img1, mtx, dist)
plot_contrast_image(img1, undistort_img1)
undistort_img2 = img_undistort(img2, mtx, dist)
plot_contrast_image(img2, undistort_img2)
undistort_img3 = img_undistort(img3, mtx, dist)
plot_contrast_image(img3, undistort_img3)
