#
#   Karlo Horvat
#   24.3.2022
#   

from tkinter import filedialog
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from matplotlib.widgets import Slider, Button
import PIL.Image, PIL.ImageTk
from tkinter import *
import copy
import tkinter
np.seterr(over='ignore')
win = Tk()
panelA = None
image_gray = None
image_gray_dark = None
nova_slika = None



def contrasPlus(image_gray):
    alpha = 1.3
    beta = 2
    
    
    for y in range(image_gray.shape[0]):
            for x in range(image_gray.shape[1]):
                image_gray[y,x] = np.clip(alpha*image_gray[y,x] + beta, 0, 255)

    return image_gray


def contras(image_gray_dark):
    alpha = 0.1
    beta = 2
    
    
    for y in range(image_gray_dark.shape[0]):
            for x in range(image_gray_dark.shape[1]):
                image_gray_dark[y,x] = np.clip(alpha*image_gray_dark[y,x] + beta, 0, 255)

    return image_gray_dark

@jit(forceobj=True)
def imageToGray(filename):
    return cv.cvtColor(cv.imread(filename),cv.COLOR_BGR2GRAY)

def getImage(filename):
    return cv.imread(filename)
#select image and convert it to grayscale 
def selectImage():
    
    global image_gray, image_gray_dark
    filename = filedialog.askopenfilename()
    
    
    #image_cv = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)
    #cv.imshow("svetlo",image)
    image_cv = imageToGray(filename)
    #spremenimo sliko v grayscale
    #image_cv = imageToGray(filename)
    #image_cv = cv.GaussianBlur(image_cv, (3,3), 0)
    #kernel = np.ones((4,4),np.float32)/25
    #image_cv = cv.filter2D(image_cv, -1, kernel)
    #image_cv = cv.medianBlur(image_cv, 3)
    #spremenimo velikost slike
    image_gray = cv.resize(image_cv, (800,600), interpolation= cv.INTER_AREA)

    #cv.imshow("Neakj",image_gray)
    image_gray = contrasPlus(image_gray)
    

    image_gray_dark = copy.deepcopy(image_gray)
    image_gray_dark = contras(image_gray_dark)

    
    image_gray_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image_gray))
    image_gray_dark_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image_gray_dark))


    canvas.create_image(0, 0, image=image_gray_tk, anchor=tkinter.NW)
    canvas.image = image_gray_tk
    
    canvas2.create_image(0, 0, image=image_gray_dark_tk, anchor=tkinter.NW)
    canvas2.image = image_gray_dark_tk
    
    #label_image= Label(win, image=image)
    #label_image.image = image
    #label_image.place(x=0, y=0, anchor="W")
    
    #win.update()
   

def sobel():
    global image_gray, image_gray_dark, nova_slika

    canvas.delete("all")
    canvas2.delete("all")
    slika = copy.deepcopy(image_gray)
    slika2 = copy.deepcopy(image_gray_dark)
    #cv.imshow("slika2", slika2)
    #nova_slika= copy.deepcopy(slika)
    #nova_slika2 = copy.deepcopy(slika2)
    #cv.imshow("slika",slika)
    visina_slika = slika.shape[0]
    sirina_slika = slika.shape[1]
    
   
    nova_slika_x = np.zeros((visina_slika, sirina_slika), dtype=np.uint8)
    nova_slika_y = np.zeros((visina_slika, sirina_slika), dtype=np.uint8)
 
    nova_slika = np.zeros((visina_slika, sirina_slika), dtype=np.uint8)
    nova_slika2 = np.zeros((visina_slika, sirina_slika), dtype=np.uint8)
    max = 1
    min = 1
    nova_slika = calcSobel(slika, nova_slika, nova_slika_x, nova_slika_y, min, max, visina_slika, sirina_slika)
    nova_slika2 = calcSobel(slika2, nova_slika2, nova_slika_x, nova_slika_y, min, max, visina_slika, sirina_slika)
   
    
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(nova_slika))
    canvas.create_image(0,0, image=image_tk, anchor=tkinter.NW)
    canvas.image = image_tk

    image_tk2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(nova_slika2))
    canvas2.create_image(0,0, image=image_tk2, anchor=tkinter.NW)
    canvas2.image = image_tk2        
    
 
    #racunski del sobela pospesen z numbo
@jit(nopython=True)
def calcSobel(slika, nova_slika, nova_slika_x, nova_slika_y, min, max, visina_slika, sirina_slika):

    for k in range(1, visina_slika - 1):
        for l in range(1, sirina_slika - 1):
            if (max < slika[k][l]):
                max = slika[k][l]

    for i in range(1, visina_slika - 1):
        for j in range(1, sirina_slika - 1):
            x = (255 / (max - min)) * (((slika[i - 1, j - 1] * (-1)) + (slika[i, j - 1] * (-2)) + (slika[i + 1, j - 1] * (-1)) + slika[i + 1, j - 1] + (slika[i, j + 1] * 2) + slika[i + 1, j + 1]) - min)
            nova_slika_x[i, j] = abs(x)
            y = (255 / (max - min)) * ((slika[i - 1, j - 1] + (slika[i - 1, j] * 2) + slika[i - 1, j + 1] + (slika[i + 1, j - 1] * (-1)) + (slika[i + 1, j] * (-2)) + (slika[i + 1, j + 1] * (-1))) - min)

            nova_slika_y[i, j] = abs(y)

            
            nova_slika[i][j] = (nova_slika_x[i, j] + nova_slika_y[i, j])

    return nova_slika
    
    

def roberts():
    canvas.delete("all")
    canvas2.delete("all")
    slika = copy.deepcopy(image_gray)
    slika2 = copy.deepcopy(image_gray_dark)
    visina_slika = slika.shape[0]
    sirina_slika = slika.shape[1]

    nova_slika_x = np.zeros((visina_slika, sirina_slika), dtype=np.uint8)
    nova_slika_y = np.zeros((visina_slika, sirina_slika), dtype=np.uint8)
    nova_slika = np.zeros((visina_slika, sirina_slika), dtype=np.uint8)
    nova_slika2 = np.zeros((visina_slika, sirina_slika), dtype=np.uint8)
    max = 1
    min = 1
    nova_slika = calcRoberts(slika, nova_slika, nova_slika_x, nova_slika_y, min, max, visina_slika, sirina_slika)
    nova_slika2 = calcRoberts(slika2, nova_slika2, nova_slika_x, nova_slika_y, min, max, visina_slika, sirina_slika)


    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(nova_slika))
    canvas.create_image(0,0, image=image_tk, anchor=tkinter.NW)
    canvas.image = image_tk

    image_tk2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(nova_slika2))
    canvas2.create_image(0,0, image=image_tk2, anchor=tkinter.NW)
    canvas2.image = image_tk2   

@jit(nopython=True)
def calcRoberts(slika, nova_slika, nova_slika_x, nova_slika_y, min, max, visina_slika, sirina_slika):
    for k in range(1, visina_slika - 1):
        for l in range(1, sirina_slika - 1):
            if (max < slika[k][l]):
                max = slika[k][l]

    for i in range(1, visina_slika - 1):
        for j in range(1, sirina_slika - 1):
            x = (255 / (max - min))*((slika[i, j] * 1) + (slika[i + 1, j + 1] * (-1))-min)
            nova_slika_x[i, j] = abs(x)

            y =  (255 / (max - min))*((slika[i, j + 1] * 1 + (slika[i + 1, j] * (-1)))-min)
            nova_slika_y[i, j] = abs(y)

            nova_slika[i][j] = nova_slika_x[i, j] + nova_slika_y[i, j]
    return nova_slika



def prewitt():
    canvas.delete("all")
    canvas2.delete("all")
    slika = copy.deepcopy(image_gray)
    slika2 = copy.deepcopy(image_gray_dark)
    visina_slika = slika.shape[0]
    sirina_slika = slika.shape[1]

    nova_slika_x = np.zeros((visina_slika, sirina_slika), dtype=np.uint8)
    nova_slika_y = np.zeros((visina_slika, sirina_slika), dtype=np.uint8)
    nova_slika = np.zeros((visina_slika, sirina_slika), dtype=np.uint8)
    nova_slika2 = np.zeros((visina_slika, sirina_slika), dtype=np.uint8)
    max = 1
    min = 1
    nova_slika = calcPrewitt(slika, nova_slika, nova_slika_x, nova_slika_y, min, max, visina_slika, sirina_slika)
    nova_slika2 = calcPrewitt(slika2, nova_slika2, nova_slika_x, nova_slika_y, min, max, visina_slika, sirina_slika)


    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(nova_slika))
    canvas.create_image(0,0, image=image_tk, anchor=tkinter.NW)
    canvas.image = image_tk

    image_tk2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(nova_slika2))
    canvas2.create_image(0,0, image=image_tk2, anchor=tkinter.NW)
    canvas2.image = image_tk2   

@jit(nopython=True)
def calcPrewitt(slika, nova_slika, nova_slika_x, nova_slika_y, min, max, visina_slika, sirina_slika):
    for k in range(1, visina_slika - 1):
        for l in range(1, sirina_slika - 1):
            if (max < slika[k][l]):
                max = slika[k][l]

    for i in range(1, visina_slika - 1):
        for j in range(1, sirina_slika - 1):
            x = (255 / (max - min)) * ((slika[i - 1, j - 1] + slika[i, j - 1] + slika[i + 1, j - 1] + (slika[i-1, j+1] * (-1)) + (slika[i, j+1] * (-1)) + (slika[i+1, j+1] * (-1))) - min)
            nova_slika_x[i, j] = abs(x)
            y = (255 / (max - min)) * ((slika[i - 1, j - 1] + slika[i - 1, j] + slika[i - 1, j + 1] + (slika[i + 1, j - 1] * (-1)) + (slika[i + 1, j] * (-1)) + (slika[i + 1, j + 1] * (-1)))-min)
            nova_slika_y[i, j] = abs(y)

            nova_slika[i][j] = nova_slika_x[i, j] + nova_slika_y[i, j]

    return nova_slika


def canny():
    canvas.delete("all")
    canvas2.delete("all")
    slika = copy.deepcopy(image_gray)
    slika2 = copy.deepcopy(image_gray_dark)

    nova_slika = cv.Canny(slika, 100,200)
    nova_slika2 = cv.Canny(slika2, 100,200)

    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(nova_slika))
    canvas.create_image(0,0, image=image_tk, anchor=tkinter.NW)
    canvas.image = image_tk

    image_tk2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(nova_slika2))
    canvas2.create_image(0,0, image=image_tk2, anchor=tkinter.NW)
    canvas2.image = image_tk2   



canvas = tkinter.Canvas(win, width = 800, height =600)
canvas.pack(side="left",padx=20, pady=20)
canvas2 = tkinter.Canvas(win, width = 800, height =600)
canvas2.pack(side="right",padx=20, pady=20)
btn = Button(win, text= "Select Image", command=selectImage)
btn.pack(side=BOTTOM)
btnSobel = Button(win, text="Sobel", command=sobel).pack(side=BOTTOM)
btnRoberts = Button(win, text="Roberts", command=roberts).pack(side=BOTTOM)
btnPrewitt = Button(win, text="Prewitt", command=prewitt).pack(side=BOTTOM)
btnCanny = Button(win, text="Canny", command=canny).pack(side=BOTTOM)
win.mainloop()


