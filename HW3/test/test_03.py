import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Fig0460a.tif', cv2.IMREAD_GRAYSCALE)
img = np.float32(img)
img = img/255

rows,cols=img.shape

rh, rl, cutoff = 3.0,0.4,20


y_log = np.log(img+0.01)

y_fft = np.fft.fft2(y_log)

y_fft_shift = np.fft.fftshift(y_fft)


DX = cols/cutoff
G = np.ones((rows,cols))
for i in range(rows):
    for j in range(cols):
        G[i][j]=((rh-rl)*(1-np.exp(-5*((i-rows/2)**2+(j-cols/2)**2)/(2*DX**2))))+rl

result_filter = G * y_fft_shift

result_interm = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))

result = np.exp(result_interm)

cv2.imshow("Homomorphic_Filter", result)
plt.imshow(result, cmap = "gray"), plt.title("Homomorphic_Filter")
plt.show()