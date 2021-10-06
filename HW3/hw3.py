###################################################################################################
#										READ ME													  #
###################################################################################################
# 1. You can only use one function at once, that is, the last process you chose/changed			  #
# 2. Please close any window that pop up before next function									  #
#																								  #
###################################################################################################
# import modules
# tkinter is already install with python / ubuntu
# in linux, use "sudo apt-get install python-tk" to intall tkinter forcibly
import tkinter as tk
from tkinter import filedialog
# Use Pillow, to install: pip3 install Pillow==7.0.0
from PIL import Image, ImageTk, ImageOps
# Use OpenCV, to install: pip3 install opencv-python==4.2.0.32
import cv2
# Use matplotlib, to install: pip3 install matplotlib==3.0.3
import matplotlib.pyplot as plt
# Use numpy, to install: pip3 install numpy==1.18.2
import numpy as np
import os, math

# Create new Window
main_window = tk.Tk()
#Set Window title
main_window.title("B073040022 張浩綸 HW3")
#Window size(width*height+x+y)
main_window.geometry("1050x800+200+50")
#Fixed size
main_window.resizable(False, False)
# Global variables
global original_img, original_img_arr, original_img_h, original_img_w, isGray

# functions
# open image using file manager
def open_image():
	# set default path to the current working directory for convenience
	file_path = filedialog.askopenfilename(initialdir = os.getcwd())
	if file_path == "":
		pass
	else:
		global original_img, original_img_h, original_img_w, original_img_arr, isGray
		original_img = Image.open(file_path)# use PIL read image instead of cv2 for convenience
		original_img_arr = np.array(Image.open(file_path))
		# original_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
		# cv2.imshow("Image", original_img) # Check
		# plt.imshow(original_img, cmap = "gray"), plt.title("Image")
		# plt.show()
		isGray = 0# a flag that know the image is grayscale or not
		# # Check if the cv2 image is gray or not, use Threshold == 1.0
		# if (len(original_img.shape) == 1) or ((original_img[0:original_img_h-1, 0:original_img_w-1, 0] - original_img[0:original_img_h-1, 0:original_img_w-1, 1] < 1.0).all() and (original_img[0:original_img_h-1, 0:original_img_w-1, 1] - original_img[0:original_img_h-1, 0:original_img_w-1, 2] < 1.0).all()):
		# 	isGray = 1
		# else:
		# 	isGray = 0

		if original_img.mode == 'RGB':
			isGray = 0
		elif original_img.mode == 'L':
			isGray = 1
		else:
			isGray = -1
		(original_img_w, original_img_h) = original_img.size
		ratio = original_img_h / original_img_w
		if ratio >= 1:
			img_show = original_img.resize((int(400/ratio), 400))
		else:
			img_show = original_img.resize((400, (int(400*ratio))))
		# Show image
		img_show_tk = ImageTk.PhotoImage(img_show)
		global label_before
		label_before.config(image = img_show_tk)
		label_before.image = img_show_tk # needed but why?

def homomorphic_filter():
	global original_img_h, original_img_w, original_img_arr
	gamma_L = float(entry_homomorphic_gammaL.get())
	gamma_H = float(entry_homomorphic_gammaH.get())
	c = float(entry_homomorphic_c.get())
	D_0 = float(entry_homomorphic_D0.get())
	# gamma_L = 0.4 # test
	# gamma_H = 3.0
	# c = 5
	# D_0 = 20
	img_arr = np.float64(original_img_arr)/255/math.exp(30)# normalize 0~1 & enhence contrast
	img_log = np.log(img_arr+0.01)# avoid ln(0)
	img_fft = np.fft.fft2(img_log, (original_img_h, original_img_w))
	img_shift_fft = np.fft.fftshift(img_fft)
	
	H = np.ones((original_img_h, original_img_w))
	for i in range(original_img_h):
		for j in range(original_img_w):
			H[i, j] = (gamma_H-gamma_L)*(1-np.exp(-c*((i-original_img_h/2)**2+(j-original_img_w/2)**2)/((3*D_0)**2))) + gamma_L# formula & some parameters added

	img_fft_fliter = H * img_shift_fft
	img_ln = np.real(np.fft.ifft2(np.fft.ifftshift(img_fft_fliter)))
	result = np.exp(img_ln)-0.01# reverse action

	# normalize 0~255
	result = ((result - np.min(result)) * (1 / (np.max(result) - np.min(result))*255)).astype("uint8")

	# # Check
	# # Open file
	# fp = open("out.txt", "w")
	# np.savetxt("out.txt", result, delimiter=', ')
	# # Close file
	# fp.close()

	# cv2.imshow("Homomorphic Filter", result) # Same result
	plt.imshow(result, cmap="gray"), plt.title("Homomorphic Filter")
	plt.show()

def R_component_img():
	global original_img_arr
	r_component_img = np.zeros((original_img_w, original_img_h, 3), dtype = "uint8")
	if isGray == 1:
		# r_component_img[ : , : ,0] = original_img_arr[ : , : ]
		r_component_img[ : , : ,0] = np.swapaxes(original_img_arr[ : , : ],0,1)
		r_component_img = np.swapaxes(r_component_img, 0, 1)
	else:
		r_component_img[ : , : ,0] = original_img_arr[ : , : ,0]
	plt.imshow(r_component_img), plt.title("R Component")
	plt.show()

def G_component_img():
	global original_img_arr
	g_component_img = np.zeros((original_img_w, original_img_h, 3), dtype = "uint8")
	if isGray == 1:
		# g_component_img[ : , : ,0] = original_img_arr[ : , : ]
		g_component_img[ : , : ,1] = np.swapaxes(original_img_arr[ : , : ],0,1)
		g_component_img = np.swapaxes(g_component_img, 0, 1)
	else:
		g_component_img[ : , : ,1] = original_img_arr[ : , : ,1]
	plt.imshow(g_component_img), plt.title("R Component")
	plt.show()

def B_component_img():
	global original_img_arr
	b_component_img = np.zeros((original_img_w, original_img_h, 3), dtype = "uint8")
	if isGray == 1:
		# b_component_img[ : , : ,0] = original_img_arr[ : , : ]
		b_component_img[ : , : ,2] = np.swapaxes(original_img_arr[ : , : ],0,1)
		b_component_img = np.swapaxes(b_component_img, 0, 1)
	else:
		b_component_img[ : , : ,2] = original_img_arr[ : , : ,2]
	plt.imshow(b_component_img), plt.title("R Component")
	plt.show()

def H_component_img():
	global original_img
	img_arr = np.array(original_img.convert("HSV"))# 0~255
	h_component_img = np.zeros((original_img_w, original_img_h), dtype = "uint8")
	if isGray == 1:
		h_component_img[ : , : ] = np.swapaxes(img_arr[ : , : , 0],0,1)
		h_component_img = np.swapaxes(h_component_img, 0, 1)
	else:
		h_component_img[ : , : ] = img_arr[ : , : , 0]
	plt.imshow(h_component_img, cmap = "gray"), plt.title("H Component")
	plt.show()

def S_component_img():
	global original_img
	img_arr = np.array(original_img.convert("HSV"))# 0~255
	s_component_img = np.zeros((original_img_w, original_img_h), dtype = "uint8")
	if isGray == 1:
		s_component_img[ : , : ] = np.swapaxes(img_arr[ : , : , 1],0,1)
		s_component_img = np.swapaxes(s_component_img, 0, 1)
	else:
		s_component_img[ : , : ] = img_arr[ : , : , 1]
	plt.imshow(s_component_img, cmap = "gray"), plt.title("S Component")
	plt.show()

def V_component_img():
	global original_img
	img_arr = np.array(original_img.convert("HSV"))# 0~255
	v_component_img = np.zeros((original_img_w, original_img_h), dtype = "uint8")
	if isGray == 1:
		v_component_img[ : , : ] = np.swapaxes(img_arr[ : , : , 2],0,1)
		v_component_img = np.swapaxes(v_component_img, 0, 1)
	else:
		v_component_img[ : , : ] = img_arr[ : , : , 2]
	plt.imshow(v_component_img, cmap = "gray"), plt.title("V Component")
	plt.show()

def color_complement():
	global original_img_arr, original_img_h, original_img_w
	# # HSV method
	# img_arr = np.array(original_img.convert("HSV"))# 0~255
	# h_component_img = np.zeros((original_img_w, original_img_h), dtype = "uint8")
	# s_component_img = np.zeros((original_img_w, original_img_h), dtype = "uint8")
	# v_component_img = np.zeros((original_img_w, original_img_h), dtype = "uint8")
	# if isGray == 1:
	# 	h_component_img[ : , : ] = np.swapaxes(img_arr[ : , : , 0],0,1)
	# 	h_component_img = np.swapaxes(h_component_img, 0, 1)
	# 	s_component_img[ : , : ] = np.swapaxes(img_arr[ : , : , 1],0,1)
	# 	s_component_img = np.swapaxes(s_component_img, 0, 1)
	# 	v_component_img[ : , : ] = np.swapaxes(img_arr[ : , : , 2],0,1)
	# 	v_component_img = np.swapaxes(v_component_img, 0, 1)
	# else:
	# 	h_component_img[ : , : ] = img_arr[ : , : , 0]
	# 	s_component_img[ : , : ] = img_arr[ : , : , 1]
	# 	v_component_img[ : , : ] = img_arr[ : , : , 2]
	# h_component_img = (h_component_img + 128) % 256
	# # s_component_img no change
	# v_component_img = 255 - v_component_img
	# img_arr = np.zeros((original_img_h, original_img_w, 3), dtype = "uint8")
	# img_arr[ : , : , 0] = h_component_img[ : , : ]
	# img_arr[ : , : , 1] = s_component_img[ : , : ]
	# img_arr[ : , : , 2] = v_component_img[ : , : ]
	# # Use RGB method
	img_arr = np.zeros((original_img_h, original_img_w, 3), dtype = "uint8")
	img_arr[ : , : , 0] = 255 - original_img_arr[ : , : , 0]
	img_arr[ : , : , 1] = 255 - original_img_arr[ : , : , 1]
	img_arr[ : , : , 2] = 255 - original_img_arr[ : , : , 2]
	plt.imshow(img_arr), plt.title("Color Complement")
	plt.show()

def smooth_and_sharpen():
	kernel_size = 5
	kernel_smooth = np.ones((kernel_size, kernel_size), dtype = "float") / (kernel_size**2)# average filter
	img_blur = cv2.filter2D(original_img_arr, -1, kernel_smooth) # use medium blur to get rid of noises
	kernel_sharpen = np.ones((kernel_size, kernel_size), dtype = "float") * -1# sharpen kernel
	kernel_sharpen[int((kernel_size+1)/2-1), int((kernel_size+1)/2-1)] = kernel_size**2
	# applying the sharpening kernel to the input image & displaying it.
	img_arr = cv2.filter2D(img_blur, -1, kernel_sharpen)
	plt.imshow(img_arr), plt.title("Smooth and Sharpen")
	plt.show()
	
def segment():# Demonstration on Hackmd: https://hackmd.io/@eRpJk3cxRRWfQYS6riaonw/rJBIcHhKU
	global original_img, original_img_arr
	img_arr = np.array(original_img.convert("HSV"))# 0~255
	purple_lower_bound = np.array([177, 40, 40])
	purple_upper_bound = np.array([212, 255, 255])
	mask = cv2.inRange(img_arr, purple_lower_bound, purple_upper_bound)# the value in range = 255, out of = 0
	purple = np.zeros_like(original_img_arr, np.uint8)
	purple[mask > 0] = original_img_arr[mask == 255]
	plt.imshow(purple), plt.title("Segment")
	plt.show()

# create & place widgets
label_original = tk.Label(main_window, text = "Original", font = ('Arial', 9))
label_original.place(x = 150, y = 20, width = 100, height = 30)

# show the original image
label_before = tk.Label(main_window, font = ('Arial', 9), bg = "#DDDDDD")
label_before.place(x = 150, y = 70, width = 400, height = 400)

button_open = tk.Button(main_window, text = "Open", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = open_image)
button_open.place(x = 20, y = 20, width = 100, height = 30)

# for Homomorphic
button_homomorphic = tk.Button(main_window, text = "Homomorphic", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = homomorphic_filter)
button_homomorphic.place(x = 20, y = 500, width = 100, height = 30)

label_homomorphic_gammaL = tk.Label(main_window, text = "GammaL", font = ('Arial', 9), fg = "black")
label_homomorphic_gammaL.place(x = 150, y = 500, width = 100, height = 30)

entry_homomorphic_gammaL = tk.Entry(main_window, bg = "#DDDDDD", font = ('Arial', 9), fg = "black")
entry_homomorphic_gammaL.place(x = 250, y = 500, width = 100, height = 30)

label_homomorphic_gammaH = tk.Label(main_window, text = "GammaH", font = ('Arial', 9), fg = "black")
label_homomorphic_gammaH.place(x = 350, y = 500, width = 100, height = 30)

entry_homomorphic_gammaH = tk.Entry(main_window, bg = "#DDDDDD", font = ('Arial', 9), fg = "black")
entry_homomorphic_gammaH.place(x = 450, y = 500, width = 100, height = 30)

label_homomorphic_c = tk.Label(main_window, text = "c", font = ('Arial', 9), fg = "black")
label_homomorphic_c.place(x = 550, y = 500, width = 100, height = 30)

entry_homomorphic_c = tk.Entry(main_window, bg = "#DDDDDD", font = ('Arial', 9), fg = "black")
entry_homomorphic_c.place(x = 650, y = 500, width = 100, height = 30)

label_homomorphic_D0 = tk.Label(main_window, text = "D_0", font = ('Arial', 9), fg = "black")
label_homomorphic_D0.place(x = 750, y = 500, width = 100, height = 30)

entry_homomorphic_D0 = tk.Entry(main_window, bg = "#DDDDDD", font = ('Arial', 9), fg = "black")
entry_homomorphic_D0.place(x = 850, y = 500, width = 100, height = 30)

button_R_component_img = tk.Button(main_window, text = "R", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = R_component_img)
button_R_component_img.place(x = 20, y = 550, width = 50, height = 30)

button_G_component_img = tk.Button(main_window, text = "G", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = G_component_img)
button_G_component_img.place(x = 70, y = 550, width = 50, height = 30)

button_B_component_img = tk.Button(main_window, text = "B", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = B_component_img)
button_B_component_img.place(x = 120, y = 550, width = 50, height = 30)

button_H_component_img = tk.Button(main_window, text = "H", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = H_component_img)
button_H_component_img.place(x = 200, y = 550, width = 50, height = 30)

button_S_component_img = tk.Button(main_window, text = "S", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = S_component_img)
button_S_component_img.place(x = 250, y = 550, width = 50, height = 30)

button_V_component_img = tk.Button(main_window, text = "V", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = V_component_img)
button_V_component_img.place(x = 300, y = 550, width = 50, height = 30)

button_color_complement = tk.Button(main_window, text = "Color Complement", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = color_complement)
button_color_complement.place(x = 380, y = 550, width = 150, height = 30)

button_smooth_and_sharpen = tk.Button(main_window, text = "Smooth and Sharpen", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = smooth_and_sharpen)
button_smooth_and_sharpen.place(x = 560, y = 550, width = 200, height = 30)

button_segment = tk.Button(main_window, text = "Segment", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = segment)
button_segment.place(x = 315, y = 600, width = 100, height = 30)

#Start
main_window.mainloop()