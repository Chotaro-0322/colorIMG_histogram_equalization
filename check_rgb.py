"pip3 install opencv-python"
"pip3 install seaborn"
"pip3 install matplotlib"
"pip3 install numpy"
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


file_name1 = "data/ex_1_night_reti.jpg"
file_name2 = "data/ex_1_noon_reti.jpg"
# file_name1 = "data/ex_1_night.JPG"
# file_name2 = "data/ex_1_noon.JPG"

print("file_name1", file_name1)
print ("file_name2", file_name2)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))

img1 = cv2.imread(file_name1)
img2 = cv2.imread(file_name2)

img1 = cv2.resize(img1, (720, 720))
img2 = cv2.resize(img2, (720, 720))

img1_gray_ori = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray_ori = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img1_gray = clahe.apply(img1_gray_ori)
img2_gray = clahe.apply(img2_gray_ori)

cv2.imshow("img1", img1_gray)
cv2.imshow("img2", img2_gray)


img1_ratio = img1_gray / img1_gray_ori
img2_ratio = img2_gray / img2_gray_ori

# print("img1_ratio ", img1_ratio)

img1_ratio_color = np.stack([img1_ratio, img1_ratio, img1_ratio], 2)
img2_ratio_color = np.stack([img2_ratio, img2_ratio, img2_ratio], 2)

img1_hist = (img1 * img1_ratio_color).astype(np.uint8)
img2_hist = (img2 * img2_ratio_color).astype(np.uint8)
# print("img1_hist ", img1_hist)

img1_gray = np.array(img1_gray, dtype=np.int16)
img2_gray = np.array(img2_gray, dtype=np.int16)

img1_cp = img1_gray.astype(np.uint8)
img2_cp = img2_gray.astype(np.uint8)

delta = np.abs(img1_gray - img2_gray).astype(np.uint8)
# delta = np.mean(delta, 2)

print ("all delta ", np.mean(delta))



cv2.imshow("img1_ori", img1)
cv2.imshow("img2_ori", img2)
cv2.imshow("img1_hist", img1_hist)
cv2.imshow("img2_hist", img2_hist)

plt.figure()
sns.heatmap(delta)
plt.savefig("./retiNit_Non.png")
plt.show()

plt.close("all")

cv2.waitKey(0)
cv2.destroyAllWindows()
