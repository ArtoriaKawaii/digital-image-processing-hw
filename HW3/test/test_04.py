import tkinter as tk
from PIL import Image ,ImageTk,ImageOps
from tkinter import filedialog
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import exp, sqrt
# TK windows basic settings
mainWindow = tk.Tk()
mainWindow.title("DIP hw1")
mainWindow.geometry("1200x800")
# make it cannot be resized
mainWindow.resizable(0,0)
global npar_for_processing
global Image_for_processing
# pass the image user want to show
def show_result(pic_to_show): 
    #make the size fit labal
    maxsize = (400, 400)
    resizePic = pic_to_show.resize(maxsize)
    global show
    show = ImageTk.PhotoImage(resizePic)
    #show on labal
    lab_imaR.config(image = show)
    lab_imaR.image = show
def openPic():
    global image_opened
    # make user choose path
    file = filedialog.askopenfilename()
    image_opened = Image.open(file)
    # save what user open as nparray
    global npar_for_processing
    npar_for_processing = np.asarray((Image.open(file)).convert("L"))
    # save for process 
    global Image_for_processing
    Image_for_processing = Image.open(file)
    # save for final when press save button
    maxsize = (400, 400)
    resizePic = image_opened.resize(maxsize)
    # show on labal after open
    global show    
    show = ImageTk.PhotoImage(resizePic)
    lab_imaL.config(image = show)
    lab_imaL.image = show
    lab_imaR.config(image = show)
    lab_imaR.image = show
def homomorphic_filter():
    d0=sca_D0.get()
    rL=sca_rL.get()
    rH=sca_rH.get()
    c=sca_c.get()
    gray = np.float64(npar_for_processing)/10**16
    height, width = gray.shape
    gray_ln = np.log(gray+0.01)  #prevent from dividing by zero encountered in log
    gray_fft = np.fft.fft2(gray_ln)
    gray_fftshift = np.fft.fftshift(gray_fft)
    gray_Huv = np.ones(gray.shape)
    Huv = np.ones(gray.shape)
    #cul D and H(u, v)
    for u in range(height):
        for v in range(width):
            d = sqrt((u - height/2.0)**2 + (v - width/2.0)**2)/d0
            Huv[u,v] = (rH - rL) * (1 - exp(-c * d**2)) + rL
    gray_Huv = gray_fftshift * Huv
    gray_ifftshift = np.fft.ifftshift(gray_Huv)
    gray_ifft = np.fft.ifft2(gray_ifftshift)
    result = np.exp(np.real(gray_ifft))
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    result = np.array(255*result, dtype = "uint8")
    show_result(Image.fromarray(result))
    # plt.imshow(result, cmap = "gray"), plt.title("Homomorphic_Filter")
    # plt.show()
def showRGBin24_bit(num_of_RGB):
    width, height = Image_for_processing.size
    pixel_values = list(Image_for_processing.getdata())
    if Image_for_processing.mode == 'RGB':
        channels = 3
    elif Image_for_processing.mode == 'L':
        channels = 1
    else:
        print("Unknown mode: %s" % Image_for_processing.mode)
        return None
    pixel_values = np.array(pixel_values).reshape((width, height, channels))
    # create zero matrix
    # 'dtype' by default: 'numpy.float64'
    split_img = np.zeros((width,height,3), dtype="uint8") 
    # assing each channel 
    split_img[ :, :, num_of_RGB] = pixel_values[ :, :, num_of_RGB]
    # display each channel
    img = Image.fromarray(split_img, 'RGB')
    show_result(img)
def toHSI(num_of_HSI):
    img_HSI = Image_for_processing.convert('HSV')
    width, height = Image_for_processing.size
    pixel_values = list(img_HSI.getdata())
    pixel_values = np.array(pixel_values).reshape((width, height, 3))
    # create zero matrix
    # 'dtype' by default: 'numpy.float64'
    split_img = np.zeros((width,height), dtype="uint8") 
    # assing each channel 
    split_img[ :, :] = pixel_values[ :, :, num_of_HSI]
    # display each channel
    img = Image.fromarray(split_img, 'L')
    show_result(img)
def Laplacian_sharp(degree):
    kernel_size = [ [ -1 for i in range(degree) ] for j in range(degree) ]
    # print(kernel_size)
    # set center to degree*degree
    kernel_size[int((degree+1)/2-1)][int((degree+1)/2-1)] = degree*degree-1
    kernel = np.array(kernel_size)
    # print(kernel)
    # applying the sharpening kernel to the input image
    sharp = cv2.filter2D(np.asarray(Image_for_processing), -1, kernel)
    # img = Image.fromarray(split_img, 'RGB')
    # img.show()
    show_result(Image.fromarray(sharp))
# buttons for  functions   open
# test for word on the button
# font for word size
# command link the button or slide to apply its function  
# bg for background color
btn1 = tk.Button(text = "open",font = 12, command = openPic)
btn1.place(x = 50,y = 50,width = 100,height = 50)
btn1.config(bg = "skyblue")
# labals for  "before" and "after"
lab_imaL = tk.Label(mainWindow, text = "image", bg = "gray", font = 24)
lab_imaL.place(x = 200 ,y = 50 ,width = 400 ,height = 400)
lab_imaR = tk.Label(mainWindow, text = "image", bg = "gray", font = 24 )
lab_imaR.place(x = 750 ,y = 50, width = 400 ,height = 400)
# labals and scale for homomorphic_filter
btn_homomorphic_filter = tk.Button(mainWindow, text = "homomorphic\nfilter", font = 24 , command = homomorphic_filter)
btn_homomorphic_filter.place(x = 50 ,y = 500, width = 125 ,height = 50)
lab_rL = tk.Label(mainWindow, text = "rL", bg = "gray", font = 24)
lab_rL.place(x = 200 ,y = 500 ,width = 50 ,height = 30)
sca_rL = tk.Scale(mainWindow, orient = "horizontal",from_=0.01,to=0.90 ,resolution = 0.01,digit = 3)
sca_rL.place(x = 250 ,y = 500, width = 250 ,height = 50)
lab_rH = tk.Label(mainWindow, text = "rH", bg = "gray", font = 24)
lab_rH.place(x = 200 ,y = 550 ,width = 50 ,height = 30)
sca_rH = tk.Scale(mainWindow, orient = "horizontal",from_=1.0,to=100.0 ,resolution = 1,digit = 3)
sca_rH.place(x = 250 ,y = 550, width = 250 ,height = 50)
lab_c = tk.Label(mainWindow, text = "c", bg = "gray", font = 24)
lab_c.place(x = 550 ,y = 500 ,width = 50 ,height = 30)
sca_c = tk.Scale(mainWindow, orient = "horizontal",from_=1,to=15 ,resolution = 1,digit = 3)
sca_c.place(x = 600 ,y = 500, width = 250 ,height = 50)
lab_D0 = tk.Label(mainWindow, text = "D0", bg = "gray", font = 24)
lab_D0.place(x = 550 ,y = 550 ,width = 50 ,height = 30)
sca_D0 = tk.Scale(mainWindow, orient = "horizontal",from_=10,to=100 ,resolution = 10,digit = 4)
sca_D0.place(x = 600 ,y = 550, width = 250 ,height = 50)
# labals and scale for RGB
btn_splitR = tk.Button(mainWindow, text = "R", font = 24 , command = lambda:showRGBin24_bit(0))
btn_splitR.place(x = 50 ,y = 600, width = 50 ,height =30)
btn_splitG = tk.Button(mainWindow, text = "G", font = 24 , command = lambda:showRGBin24_bit(1))
btn_splitG.place(x = 100 ,y = 600, width = 50 ,height = 30)
btn_splitB = tk.Button(mainWindow, text = "B", font = 24 , command = lambda:showRGBin24_bit(2))
btn_splitB.place(x = 150 ,y = 600, width = 50 ,height = 30)
# labals and scale for HSI
btn_splitH = tk.Button(mainWindow, text = "H", font = 24 , command = lambda:toHSI(0))
btn_splitH.place(x = 250 ,y = 600, width = 50 ,height = 30)
btn_splitS = tk.Button(mainWindow, text = "S", font = 24 , command = lambda:toHSI(1))
btn_splitS.place(x = 300 ,y = 600, width = 50 ,height = 30)
btn_splitI = tk.Button(mainWindow, text = "I", font = 24 , command = lambda:toHSI(2))
btn_splitI.place(x = 350 ,y = 600, width = 50 ,height = 30)
kernel_size = 5
btn_splitH = tk.Button(mainWindow, text = "Laplacian_sharp", font = 24 , command = lambda:Laplacian_sharp(kernel_size))
btn_splitH.place(x = 450 ,y = 600, width = 100 ,height = 30)
#keep the mainwindow stay on the screen
mainWindow.tk.mainloop()