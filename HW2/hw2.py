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
import numpy, os

# Create new Window
main_window = tk.Tk()
#Set Window title
main_window.title("B073040022 張浩綸 HW2")
#Window size(width*height+x+y)
main_window.geometry("1050x800+200+50")
#Fixed size
main_window.resizable(False, False)
# Global variables
global original_img, original_img_row, original_img_col

# functions
def show_processed_img(img_cv2):
	img_cv2_re = cv2.resize(img_cv2, (400, 400))
	img_show_pil = Image.fromarray(img_cv2_re)
	img_show_tk = ImageTk.PhotoImage(img_show_pil)
	global label_after # no need, already global
	label_after.config(image = img_show_tk)
	label_after.image = img_show_tk  # needed but why?

# open image using file manager
def open_image():
	# set default path to the current working directory for convenience
	file_path = filedialog.askopenfilename(initialdir = os.getcwd())
	if file_path == "":
		pass
	else:
		global original_img, original_img_row, original_img_col
		original_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
		(original_img_row, original_img_col) = original_img.shape
		img_show = cv2.resize(original_img, (400, 400))
		img_show_pil = Image.fromarray(img_show)
		img_show_tk = ImageTk.PhotoImage(img_show_pil)
		global label_before # no need, already global
		label_before.config(image = img_show_tk)
		label_before.image = img_show_tk # needed but why?
		show_processed_img(img_show)

def gray_level_slicing():
	global entry_unselected_area_lower_bound, entry_unselected_area_upper_bound
	global original_img, original_img_row, original_img_col
	global listbox_unselected_area
	mode = int(listbox_unselected_area.curselection()[0]) # 0: preserve, 1: black
	img_show = numpy.zeros((original_img_row,original_img_col),dtype = "uint8")
	# get entries value and store them as ints
	lower_bound = int(entry_unselected_area_lower_bound.get())
	upper_bound = int(entry_unselected_area_upper_bound.get())
	if mode == 0:# preserve
		for i in range(original_img_row):
			for j in range(original_img_col):
				if original_img[i, j] > lower_bound and original_img[i, j] < upper_bound:
					img_show[i, j] = 255# set area white
				else:
					img_show[i, j] = int(original_img[i, j])# preserve
	elif mode == 1:# black
		for i in range(original_img_row):
			for j in range(original_img_col):
				if original_img[i, j] > lower_bound and original_img[i, j] < upper_bound:
					img_show[i, j] = 255# set area white
				else:
					img_show[i, j] = 0# black
	else:
		print("Error!\n")
	show_processed_img(img_show)

def bit_plane_image():
	global scale_bit_plane, original_img
	bit = int(scale_bit_plane.get())
	img = original_img.copy()
	img[img / (2**bit) % 2 >= 1] = 255
	img[img / (2**bit) % 2 < 1] = 0
	show_processed_img(img)

scale_smoothing_kernal_size_fix_var = 1
def scale_smoothing_kernal_size_fix(n): # helping func.
	global scale_smoothing_kernal_size_fix_var, scale_smoothing_kernal_size
	n = int(n)
	if not n % 2:
		# if n > scale_smoothing_kernal_size_fix_var: n + 1
		# else : n-1
		scale_smoothing_kernal_size.set(n+1 if n > scale_smoothing_kernal_size_fix_var else n-1) # need explain!!
		scale_smoothing_kernal_size_fix_var = scale_smoothing_kernal_size.get()

def smoothing():
	global scale_smoothing_kernal_size, original_img
	img = original_img.copy()
	kernal_size = int(scale_smoothing_kernal_size.get())
	img = cv2.medianBlur(img, kernal_size) # use medium blur to get rid of noises
	show_processed_img(img)

scale_sharpening_kernal_size_fix_var = 1
def scale_sharpening_kernal_size_fix(n): # helping func.
	global scale_sharpening_kernal_size_fix_var, scale_sharpening_kernal_size
	n = int(n)
	if not n % 2:
		# if n > scale_sharpening_kernal_size_fix_var: n + 1
		# else : n-1
		scale_sharpening_kernal_size.set(n+1 if n > scale_sharpening_kernal_size_fix_var else n-1) # need explain!!
		scale_sharpening_kernal_size_fix_var = scale_sharpening_kernal_size.get()

def sharpening():
	global scale_sharpening_kernal_size, original_img
	img = original_img.copy()
	kernal_size = int(scale_sharpening_kernal_size.get())
	if kernal_size == 1:
		pass
	else:
		# Create our shapening kernel, it must equal to one eventually
		kernel_sharpening = numpy.ones((kernal_size, kernal_size), dtype = int) * -1
		kernel_sharpening[int((kernal_size+1)/2), int((kernal_size+1)/2)] = kernal_size*kernal_size
		# applying the sharpening kernel to the input image & displaying it.
		img = cv2.filter2D(img, -1, kernel_sharpening)
	show_processed_img(img)

def fft():
	global original_img
	fft = numpy.fft.fft2(original_img) # Fourier Transform
	fft_shift = numpy.fft.fftshift(fft) # Shift
	spectrum = numpy.log(1+numpy.abs(fft_shift)) # c*log|1+F(u,v)|
	plt.imshow(spectrum, cmap = "gray"), plt.title("Spectrum")
	plt.show()
	

def amplitude():
	global original_img
	fft = numpy.fft.fft2(original_img) # Fourier Transform
	amplitude_spec = numpy.abs(fft)# get the amplituded spectrum of original_img
	amplitude_img = numpy.log(numpy.abs(numpy.fft.ifft2(amplitude_spec))) # Inverse Fourier Transform
	plt.imshow(amplitude_img, cmap = "gray"), plt.title("Amplitude Image")
	plt.show()

def phase():
	global original_img
	fft = numpy.fft.fft2(original_img) # Fourier Transform
	phase_spec = fft / numpy.abs(fft) # get the phased spectrum of original_img
	# phase_spec = numpy.angle(fft) # have mirror image
	phase_img = numpy.abs(numpy.fft.ifft2(phase_spec)) # Inverse Fourier Transform
	plt.imshow(phase_img, cmap = "gray"), plt.title("Phase Image")
	plt.show()

# create & place widgets
label_original = tk.Label(main_window, text = "Original", font = ('Arial', 9))
label_original.place(x = 150, y = 20, width = 100, height = 30)

label_processed = tk.Label(main_window, text = "Processed", font = ('Arial', 9))
label_processed.place(x = 600, y = 20, width = 100, height = 30)

# show the original image
label_before = tk.Label(main_window, font = ('Arial', 9), bg = "#DDDDDD")
label_before.place(x = 150, y = 70, width = 400, height = 400)

# show the image after last process(only last one)
label_after = tk.Label(main_window, font = ('Arial', 9), bg = "#DDDDDD")
label_after.place(x = 600, y = 70, width = 400, height = 400)

button_open = tk.Button(main_window, text = "Open", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = open_image)
button_open.place(x = 20, y = 20, width = 100, height = 30)

label_method = tk.Label(main_window, text = "Method", font = ('Arial', 9), fg = "black")
label_method.place(x = 20, y = 430, width = 100, height = 30)

# for Gray-level Slicing
label_gray_level_slicing = tk.Label(main_window, text = "Gray-level\nSlicing", font = ('Arial', 9), fg = "black")
label_gray_level_slicing.place(x = 20, y = 490, width = 100, height = 40)

label_unselected_area = tk.Label(main_window, text = "Unselected\nArea", font = ('Arial', 9), fg = "black")
label_unselected_area.place(x = 120, y = 490, width = 100, height = 40)

listbox_unselected_area = tk.Listbox(main_window, bg = "#DDDDDD", font = ('Arial', 9), fg = "black")
listbox_unselected_area.insert(0, "Preserve")
listbox_unselected_area.insert(1, "Black")
listbox_unselected_area.place(x = 220, y = 495, width = 60, height = 40)

label_unselected_area_lower_bound = tk.Label(main_window, text = "Lower\nBound", font = ('Arial', 9), fg = "black")
label_unselected_area_lower_bound.place(x = 280, y = 490, width = 70, height = 40)

entry_unselected_area_lower_bound = tk.Entry(main_window, bg = "#DDDDDD", font = ('Arial', 9), fg = "black")
entry_unselected_area_lower_bound.place(x = 350, y = 495, width = 50, height = 30)

label_unselected_area_upper_bound = tk.Label(main_window, text = "Upper\nBound", font = ('Arial', 9), fg = "black")
label_unselected_area_upper_bound.place(x = 400, y = 490, width = 70, height = 40)

entry_unselected_area_upper_bound = tk.Entry(main_window, bg = "#DDDDDD", font = ('Arial', 9), fg = "black")
entry_unselected_area_upper_bound.place(x = 470, y = 495, width = 50, height = 30)

button_gray_level_slicing_apply = tk.Button(main_window, text = "Apply", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = gray_level_slicing)
button_gray_level_slicing_apply.place(x = 550, y = 495, width = 100, height = 30)

# for bit-plane images
label_bit_plane = tk.Label(main_window, text = "Bit-Plane", font = ('Arial', 9), fg = "black")
label_bit_plane.place(x = 20, y = 550, width = 100, height = 30)

scale_bit_plane = tk.Scale(main_window, from_ = 0, to = 7, resolution = 1, orient="horizontal")
scale_bit_plane.place(x = 120, y = 535, width = 300, height = 50)

botton_bit_plane_apply = tk.Button(main_window, text = "Apply", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = bit_plane_image)
botton_bit_plane_apply.place(x = 440, y = 550, width = 100, height = 30)

# for smoothing
label_smoothing = tk.Label(main_window, text = "Smoothing", font = ('Arial', 9), fg = "black")
label_smoothing.place(x = 20, y = 600, width = 100, height = 30)

label_smoothing_kernal_size = tk.Label(main_window, text = "Kernal size(n x n)", font = ('Arial', 9), fg = "black")
label_smoothing_kernal_size.place(x = 130, y = 600, width = 100, height = 30)

scale_smoothing_kernal_size = tk.Scale(main_window, from_ = 1, to = 31, command = scale_smoothing_kernal_size_fix, orient="horizontal")
scale_smoothing_kernal_size.place(x = 250, y = 590, width = 600, height = 50)

button_smoothing_apply = tk.Button(main_window, text = "Apply", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = smoothing)
button_smoothing_apply.place(x = 870, y = 600, width = 100, height = 30)

# for sharpening
label_sharpening = tk.Label(main_window, text = "Sharpening", font = ('Arial', 9), fg = "black")
label_sharpening.place(x = 20, y = 650, width = 100, height = 30)

label_sharpening_kernal_size = tk.Label(main_window, text = "Kernal size(n x n)", font = ('Arial', 9), fg = "black")
label_sharpening_kernal_size.place(x = 130, y = 650, width = 100, height = 30)

scale_sharpening_kernal_size = tk.Scale(main_window, from_ = 1, to = 31, command = scale_sharpening_kernal_size_fix, orient="horizontal")
scale_sharpening_kernal_size.place(x = 250, y = 640, width = 600, height = 50)

button_sharpening_apply = tk.Button(main_window, text = "Apply", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = sharpening)
button_sharpening_apply.place(x = 870, y = 650, width = 100, height = 30)

# for FFT
button_fft = tk.Button(main_window, text = "Fourier Transform", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = fft)
button_fft.place(x = 20, y = 700, width = 150, height = 30)

# for amplitude
button_amplitude = tk.Button(main_window, text = "Amplitude", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = amplitude)
button_amplitude.place(x = 200, y = 700, width = 150, height = 30)

# for phase
button_phase = tk.Button(main_window, text = "Phase", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = phase)
button_phase.place(x = 380, y = 700, width = 150, height = 30)

#Start
main_window.mainloop()