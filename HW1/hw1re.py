# import modules
# tkinter is already install with python / ubuntu
# in linux, use "sudo apt-get install python-tk" to intall tkinter forcibly
import tkinter as tk
from tkinter import filedialog
# Use Pillow, to install: pip install Pillow
from PIL import Image, ImageTk, ImageOps
# Use matplotlib, to install: pip install matplotlib
import matplotlib.pyplot as plot
import numpy
# Create new Window
main_window = tk.Tk()
#Set Window title
main_window.title("Image process application")
#Window size(width*height+x+y)
main_window.geometry("1050x690+200+50")
#Fixed size
main_window.resizable(False, False)
# Global variables
global original_img
global original_img_w, original_img_h
global original_img_array
global original_img_show_array
global original_img_show
global scale_ratio
global method # 0 for linear, 1 for exponential, 2 for logarithmical
global method_a
global method_b
# functions
# change the image under the "Processed" label
def show_image(img):
	show_img = ImageTk.PhotoImage(img)
	global label_after # no need, already global
	label_after.config(image = show_img)
	label_after.image = show_img  # needed but why?

def scale_zoom_operation(scale_var):
	global scale_ratio
	scale_ratio = float(scale_var)*0.01
	global original_img_show
	(original_img_show_w, original_img_show_h) = original_img_show.size
	img_temp = original_img_show.resize((int(original_img_show_w*scale_ratio), int(original_img_show_h*scale_ratio)), Image.ANTIALIAS)
	show_process_image(img_temp)

# open image using file manager
def open_image():
	file_path = filedialog.askopenfilename()
	if file_path == "":
		pass
	else:
		global original_img, original_img_show, original_img_array, original_img_show_array, original_img_w, original_img_h
		original_img = Image.open(file_path)
		(original_img_w, original_img_h) = original_img.size 
		original_img_show = Image.open(file_path)
		original_img_show = original_img_show.resize((400, 400), Image.ANTIALIAS)
		# Save Image as Numpy Array consist of pixel's
		original_img_array = numpy.array(original_img)
		original_img_show_array = numpy.array(original_img_show)
		show_img = ImageTk.PhotoImage(original_img_show)
		global label_before # no need, already global
		label_before.config(image = show_img)
		label_before.image = show_img # needed but why?
		global label_after # no need, already global
		label_after.config(image = show_img)
		label_after.image = show_img  # needed but why?
		# initialize
		global scale_zoom, histogram_times
		scale_zoom.set(100)

# apply all changes to original_img and save
def save_image():
	global original_img
	original_img.convert("L").save(filedialog.asksaveasfilename())

# histogram
def histogram_equalize():
	global original_img
	plot.figure(num = "Histogram")
	temp_histogram_array = numpy.array(original_img.histogram())
	plot.bar(numpy.arange(len(temp_histogram_array)),temp_histogram_array)
	original_img = ImageOps.equalize(original_img.convert("L"), mask = None)
	show_image(original_img)
	plot.show()

# Methods
def linear(a, b):# ax+b
	global method_a, method_b, original_img_show_array, original_img_show
	method_a = a
	method_b = b
	temp_show_array = original_img_show_array * float(a) + float(b)
	original_img_show_array[original_img_show_array > 255] = 255
	original_img_show = Image.fromarray(temp_show_array).convert("L")
	show_process_image(original_img_show)

def exponential(a, b):# exp(ax+b)
	global method_a, method_b, original_img_show_array
	method_a = a
	method_b = b
	temp_show_array = numpy.exp(original_img_show_array * float(a) + float(b))
	original_img_show_array[original_img_show_array > 255] = 255
	original_img_show = Image.fromarray(temp_show_array).convert("L")
	show_process_image(original_img_show)

def logarithmical(a, b):# ln(ax+b), b > 1
	global method_a, method_b, original_img_show_array
	method_a = a
	method_b = b
	if method_b < 1 :
		method_b = 1
	temp_show_array = numpy.log(original_img_show_array * float(a) + float(b))
	original_img_show_array[original_img_show_array > 255] = 255
	original_img_show = Image.fromarray(temp_show_array).convert("L")
	show_process_image(original_img_show)

def choose_method():
	global method, original_img_show, original_img
	original_img_show = original_img
	method = combobox_method.current()
	if method == 0:
		linear(scale_method_a.get(),scale_method_b.get())
	elif method == 1:
		exponential(scale_method_a.get(),scale_method_b.get())
	elif method == 2:
		logarithmical(scale_method_a.get(),scale_method_b.get())
	else:
		print("Select method first")

# create & place widgets
label_original = tk.Label(main_window, text = "Original", font = ('Arial', 9))
label_original.place(x = 150, y = 20, width = 100, height = 30)

label_processed = tk.Label(main_window, text = "Processed", font = ('Arial', 9))
label_processed.place(x = 600, y = 20, width = 100, height = 30)

button_open = tk.Button(main_window, text = "Open", font = ('Arial', 9), bg = "#DDDDDD", fg = "black", command = open_image)
button_open.place(x = 20, y = 20, width = 100, height = 30)

button_save = tk.Button(main_window, text = "Save", font = ('Arial', 9), bg = "#DDDDDD", command = save_image)
button_save.place(x = 20, y = 70, width = 100, height = 30)

# show the original image
label_before = tk.Label(main_window, font = ('Arial', 9), bg = "#DDDDDD")
label_before.place(x = 150, y = 70, width = 400, height = 400)

# show the image after last process(only last one)
label_after = tk.Label(main_window, font = ('Arial', 9), bg = "#DDDDDD")
label_after.place(x = 600, y = 70, width = 400, height = 400)

label_method = tk.Label(main_window, text = "Method", font = ('Arial', 9), fg = "black")
label_method.place(x = 20, y = 430, width = 100, height = 30)

listbox_method = tk.Listbox(main_window, bg = "#DDDDDD", font = ('Arial', 9), fg = "black")
listbox_method.insert(1, "Linear")
listbox_method.insert(2, "Exponential")
listbox_method.insert(3, "Logarithmical")
listbox_method.place(x = 20, y = 450, width = 100, height = 90)

label_method_a = tk.Label(main_window, text = "a", font = ('Arial', 9), fg = "black")
label_method_a.place(x = 150, y = 490, width = 50, height = 30)

scale_method_a = tk.Scale(main_window, from_ = 0, to = 10, resolution = 0.1, orient="horizontal")
scale_method_a.place(x = 200, y = 475, width = 800)

label_method_b = tk.Label(main_window, text = "b", font = ('Arial', 9), fg = "black")
label_method_b.place(x = 150, y = 550, width = 50, height = 30)

scale_method_b = tk.Scale(main_window, from_ = -100, to = 100, orient="horizontal")
scale_method_b.place(x = 200, y = 535, width = 800)

button_histogram = tk.Button(main_window, text = "Histogram", font = ('Arial', 9), bg = "#DDDDDD", command = histogram_equalize)
button_histogram.place(x = 20, y = 120, width = 100, height = 30)

#Start
main_window.mainloop()