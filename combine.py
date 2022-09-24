import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# 读入待拼接的两张彩色原始图1.jpg和2.jpg
img_1 = cv2.imread('1.jpg', 1)
img_2 = cv2.imread('2.jpg', 1)

'''
#orb算法 提取出两张图的关键点和特征向量
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img_1, None)
kp2, des2 = orb.detectAndCompute(img_2, None)
'''

# sift算法 提取出两张图的关键点和特征向量
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img_1, None)
kp2, des2 = sift.detectAndCompute(img_2, None)

# 筛选出合适的匹配索引值
bf = cv2.BFMatcher.create()
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
# print(matches)

Good_point_lmt = 0.99
goodpoints = []
for i in range(len(matches)-1):
    if matches[i].distance < Good_point_lmt * matches[i+1].distance:
        goodpoints.append(matches[i])
# print(goodpoints)

# 对1.jpg和2.jpg进行匹配
img_3 = cv2.drawMatches(img_1, kp1, img_2, kp2, goodpoints, flags=2, outImg=None)

# 计算多个二维点对之间的最优单映射变换矩阵
pts1 = np.float32([kp1[m.queryIdx].pt for m in goodpoints]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in goodpoints]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(pts2, pts1, cv2.RHO)

h1, w1, p1 = img_2.shape
h2, w2, p2 = img_1.shape
h, w = np.maximum(h1, h2), np.maximum(w1, w2)

_movedits = int(np.maximum(pts2[0][0][0], pts1[0][0][0]))
img_trans = cv2.warpPerspective(img_2, M, (w1+w2-_movedits, h))

M1 = np.float32([[1, 0, 0], [0, 1, 0]])
h_1, w_1, p = img_1.shape
dst1 = cv2.warpAffine(img_1, M1, (w1+w2-_movedits, h))

dst = cv2.add(dst1, img_trans)
dst_no = np.copy(dst)
dst_target = np.maximum(dst1, img_trans)

cv2.imshow("feature.jpg", img_3)
cv2.waitKey(0)
cv2.imshow("out_put.jpg", dst_target)
cv2.waitKey(0)

# 根据1.jpg和2.jpg的形状和大小进行拼接具体过程
fig = plt.figure(tight_layout=True, figsize=(8, 18))
gs = gridspec.GridSpec(5, 2)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(img_1)
ax = fig.add_subplot(gs[0, 1])
ax.imshow(img_2)
ax = fig.add_subplot(gs[1, :])
ax.imshow(img_3)
ax = fig.add_subplot(gs[2, 0])
ax.imshow(dst1)
ax = fig.add_subplot(gs[2, 1])
ax.imshow(img_trans)
ax = fig.add_subplot(gs[3, :])
ax.imshow(dst_no)
ax = fig.add_subplot(gs[4, :])
ax.imshow(dst_target)
plt.show()
