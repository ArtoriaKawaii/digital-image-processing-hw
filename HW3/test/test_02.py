from tkinter import*
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageEnhance, ImageOps, ImageTk
from matplotlib import pyplot as plt
import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt

#Global variables
prestimg="" #Global variable of the image location
img = Image.new('RGB', (1,1)) #Global variable storing PIL object
aftimg = Image.new('RGB', (1,1))#Global variable storing PIL object
HSI_image = Image.new('HSV',(1,1))
filename="" #filename variable
width=100 #Target image width
height=100 #Target image height
smooth_factor=1 
sharpen_factor=1

def fourier():
    global prestimg
    open_cv_image = np.array(prestimg)
    # open_cv_image = np.log(open_cv_image+0.01)
    f = np.fft.fft2(open_cv_image) #2D-FFT
    fshift = np.fft.fftshift(f) #Shift the output of fft
    # magnitude_spectrum = np.log(np.abs(fshift)) #getting the spectrum
    D = np.zeros((1162, 746))
    print(fshift.shape) #(1162, 746).
    for i in range(fshift.shape[0]):
        for j in range(fshift.shape[1]):
            D[i, j] = ((i-fshift.shape[0]/2)**2+(j-fshift.shape[1]/2)**2)**0.5 #Getting D0
    g_H = 3.0
    g_L = 0.4
    c = 5
    c = c*(-1)
    D0 = 20
    H = np.zeros((1162, 746))
    for i in range(fshift.shape[0]):
        for j in range(fshift.shape[1]):
            H[i,j] = (g_H - g_L)*(1-math.exp(c*(D[i, j]**2)/D0**2))+g_L #Making the H part.
            fshift[i,j] = fshift[i,j]*H[i,j] #Multiplying in frequency domain.
    output = np.fft.ifftshift(fshift)
    output = np.real(np.fft.ifft2(output))
    # output = np.exp(output)
    plt.subplot(121),plt.imshow(open_cv_image, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(output, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

def red():
    global filename
    temp = Image.open(filename)
    data = temp.getdata()
    #Suppress specific bands (e.g. (255, 120, 65) -> (0, 120, 0) for g)
    r = [(d[0], 0, 0) for d in data]
    temp.putdata(r)
    temp = Myresize(temp)
    temp = ImageTk.PhotoImage(temp) #Transforming it to TK-readable type.
    global label_aft #Making the function use global variable
    label_aft.config(image = temp) #Setting tkinter element
    label_aft.image=temp #Setting tkinter element

def green():
    global filename
    temp = Image.open(filename)
    data = temp.getdata()
    #Suppress specific bands (e.g. (255, 120, 65) -> (0, 120, 0) for g)
    g = [(0, d[1], 0) for d in data]
    temp.putdata(g)
    temp = Myresize(temp)
    temp = ImageTk.PhotoImage(temp) #Transforming it to TK-readable type.
    global label_aft #Making the function use global variable
    label_aft.config(image = temp) #Setting tkinter element
    label_aft.image=temp #Setting tkinter element

def blue():
    global filename
    temp = Image.open(filename)
    data = temp.getdata()
    #Suppress specific bands (e.g. (255, 120, 65) -> (0, 120, 0) for g)
    b = [(0, 0, d[2]) for d in data]
    temp.putdata(b)
    temp = Myresize(temp)
    temp = ImageTk.PhotoImage(temp) #Transforming it to TK-readable type.
    global label_aft #Making the function use global variable
    label_aft.config(image = temp) #Setting tkinter element
    label_aft.image=temp #Setting tkinter element

def display_Hue():
    global HSI_image
    Hue, Sat, Int = HSI_image.split() #Getting each channel
    plt.imshow(np.asarray(Hue),cmap='gray')
    plt.show()

def display_Sat():
    global HSI_image
    Hue, Sat, Int = HSI_image.split() #Getting each channel
    plt.imshow(np.asarray(Sat),cmap='gray')
    plt.show()

def display_Int():
    global HSI_image
    Hue, Sat, Int = HSI_image.split() #Getting each channel
    plt.imshow(np.asarray(Int),cmap='gray')
    plt.show()

def Myresize(img):
    #Making the image fits the present box.
	w_box = 450
	h_box = 600
	w, h = img.size
	f1 = 1.0*w_box/w   
	f2 = 1.0*h_box/h
	ratio = min([f1, f2])  
	width = int(w*ratio) #adjusting
	height = int(h*ratio) #adjusting
	return img.resize((width, height), Image.ANTIALIAS) #resize image

def color_complements():
    global filename
    temp = Image.open(filename)
    w, h = temp.size
    for j in range(h):
        for i in range(w):
            rgba = temp.getpixel((i, j))
            rgba = (255-rgba[0], 255-rgba[1], 255-rgba[2])
            temp.putpixel((i,j), rgba)
    render = ImageTk.PhotoImage(temp)
    global label_aft #Making the function use global variable
    label_aft.config(image = render) #Setting tkinter element
    label_aft.image=render #Setting tkinter element

def save():
    global height,width,aftimg
    temp = aftimg.resize((width,height),Image.BILINEAR) #Interpolation
    filename = filedialog.asksaveasfilename()
    temp.save(filename) #saving...

def Opening(event=None):
    global prestimg,aftimg,img,filename,HSI_image #Making the function use global variable
    filename = filedialog.askopenfilename() #To ask the user choosing the desire pic.
    prestimg = Image.open(filename) #Opening image using PIL.
    HSI_image = prestimg.convert("HSV")
    show = Myresize(prestimg) #Resizing image to fit the present box.
    img = ImageTk.PhotoImage(show) #Transforming it to TK-readable type.
    global label_aft #Making the function use global variable
    label_aft.config(image = img) #Setting tkinter element
    label_aft.image=img #Setting tkinter element

def editing():
    global filename
    temp = cv.imread(filename)
    kernel = np.ones((5,5),np.float32)/25 #Smoothing Kernal
    kernel=np.asarray(kernel)

    sharpen_factor = 5 #Setting the Laplacian Kernel size
    kernel_shapen=[[-1 for i in range(sharpen_factor)] for j in range(sharpen_factor)]
    kernel_shapen[int((sharpen_factor+1)/2-1)][int((sharpen_factor+1)/2-1)] = sharpen_factor*sharpen_factor
    kernel_shapen = np.array(kernel_shapen)

    temp = cv.filter2D(temp,-1,kernel)
    temp = cv.filter2D(temp,-1,kernel_shapen) #Sharpen

    plt.subplot(121),plt.imshow(cv.imread(filename)[:, :, [2, 1, 0]])
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(temp[:, :, [2, 1, 0]])
    plt.title('Ouput Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def segment():
    global filename
    temp = Image.open(filename)
    plt = cv.cvtColor(np.asarray(temp),cv.COLOR_RGB2BGR)
    HSV = cv.cvtColor(plt, cv.COLOR_BGR2HSV)

    # 設定一段顏色範圍
    BottomPurple = np.array([125, 43, 46]) #Upper limit
    UpperPurple = np.array([155, 255, 255]) #Lower limit

    # 透過Mask來判斷圖片某一點有沒有在該範圍裏面
    mask = cv.inRange(HSV, BottomPurple, UpperPurple)

    # 再利用類似邏輯AND來取出那些像素
    Purple_Things = cv.bitwise_and(plt, plt, mask = mask)
    Purple_Things = cv.cvtColor(Purple_Things, cv.COLOR_HSV2BGR)
    plt = Image.fromarray(cv.cvtColor(Purple_Things,cv.COLOR_BGR2RGB))
    plt.show()

win = Tk() #Creating a tkinter windowwin.title("DIP HW1") #Setting window title
win.geometry("1500x900+200+40") #Setting window size
win.resizable(0,0) #Making the app not resizeable

open = Button(win, text='Open', command=Opening,width=10,height=2) #Creating button for opening the image
save = Button(win, text="Save", command=save,width=10,height=2) #Creating button for saveing the image
open.place(x=40, y=50, anchor='nw') #Placing the button
save.place(x=40, y=125, anchor='nw') #Placing the button
red_btn = Button(win, text="Red", command=red,width=10,height=2)
red_btn.place(x=700,y=150, anchor='nw')
green_btn = Button(win, text="Green", command=green,width=10,height=2)
green_btn.place(x=700,y=200, anchor='nw')
blue_btn = Button(win, text="Blue", command=blue,width=10,height=2)
blue_btn.place(x=700,y=250, anchor='nw')

Hue_btn = Button(win, text="Hue", command=display_Hue,width=10,height=2)
Hue_btn.place(x=700,y=300, anchor='nw')
Sat_btn = Button(win, text="Sat", command=display_Sat,width=10,height=2)
Sat_btn.place(x=700,y=350, anchor='nw')
Int_btn = Button(win, text="Int", command=display_Int,width=10,height=2)
Int_btn.place(x=700,y=400, anchor='nw')
comp = Button(win, text="Complement", command=color_complements,width=10,height=2)
comp.place(x=700,y=580, anchor='nw')
seg = Button(win, text="Feathers", command=segment,width=10,height=2)
seg.place(x=700,y=530, anchor='nw')

edit_btn = Button(win, text="Filter \n & \n Sharpen", command=editing,width=10,height=4)
edit_btn.place(x=700,y=450, anchor='nw')

ft = Button(win, text='Filter!', command=fourier,width=10,height=2)
ft.place(x=250, y=680, anchor='nw') #Placing the button

g_H = Entry(width=10)
g_H.place(x=250, y=200)
g_H.insert(END, '0')

g_L = Entry(width=10)
g_L.place(x=250, y=350)
g_L.insert(END, '255')

c = Entry(width=10)
c.place(x=250, y=500)
c.insert(END, '255')

D0 = Entry(width=10)
D0.place(x=250, y=650)
D0.insert(END, '255')

lbl2_text=Label(win, text = "Image", bg = "#dbfffd",font=('Arial',18))#Creating label for after
lbl2_text.place(x=1075, y=10)

warn=Label(win, text = "Warning\n These functions are independent.", bg = "gray",font=('Arial',8))#Creating label for after
warn.place(x=700, y=100)

label_aft = Label(win, text = "Image", bg = "#DDDDDD")
label_aft.place(x = 925, y = 50, width = 450, height = 600)

g_H_label = Label(win, text = "Gamma (H)", bg = "#DDDDDD")
g_H_label.place(x = 250, y = 150, width = 140, height = 40)

g_L_label = Label(win, text = "Gamma (L)", bg = "#DDDDDD")
g_L_label.place(x = 250, y = 300, width = 140, height = 40)

c_label = Label(win, text = "C", bg = "#DDDDDD")
c_label.place(x = 250, y = 450, width = 140, height = 40)

D0_label = Label(win, text = "D0", bg = "#DDDDDD")
D0_label.place(x = 250, y = 600, width = 140, height = 40)

Opening()
win.mainloop()