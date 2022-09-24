import cv2
import matplotlib.pyplot as plt

# 读取模板
face = cv2.imread('face.jpg', 0)

# 读取图片
img = cv2.imread('Napoleon.jpg', 0)

# 获取到模板的大小h,w
h, w = face.shape[:2]

# 开始模板匹配（经过对比发现标准平方差匹配效果最好）
res = cv2.matchTemplate(img, face, 1)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# 画出检测到的部分
imgcpy = img.copy()
img = cv2.rectangle(imgcpy, top_left, bottom_right, 255, 2)
cv2.imshow('img', img)
cv2.waitKey(0)

plt.subplot(121)
plt.imshow(res, cmap='gray')
plt.subplot(122)
plt.imshow(imgcpy, cmap='gray')
plt.show()
