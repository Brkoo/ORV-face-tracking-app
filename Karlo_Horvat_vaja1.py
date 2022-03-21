from tkinter import W
from turtle import width
import cv2 as cv
import numpy as np
from numba import jit
from numba import njit
#import matplotlib.pyplot as plt


red = 170
green = 110
blue = 95


cap = cv.VideoCapture("IMG_2553.mp4")

width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

print("Dimenziej so {} x {}".format(width, height))

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

#change_res(320, 240)

bestcoordinates = (0, 0)


@jit(nopython=True)
def moveRectangle(image):
    numberOfMatches = 0
    bestcoordinates = (0,0)
    h, w, c = image.shape
    for i in range(0,h, 17):
     for j in range(0,w, 14):
        pt1 = (i, j)
        pt2 = (pt1[0]+20, pt1[1]+20)
        counter = 0
        for x in range(i, i+85):
            for y in range(j, j+70):
                pixelcolorBlue = image[x,y,0]
                pixelcolorGreen = image[x,y,1]
                pixelcolorRed = image[x,y,2]
                if pixelcolorBlue in range(blue-5, blue+5) and pixelcolorGreen in range(green-5, green+5) and pixelcolorRed in range(red-5, red+5):
                    counter += 1
        if numberOfMatches < counter:
            bestcoordinates = (i, j)
            numberOfMatches = counter

         #point.append([pt1,pt2])
       # cv.rectangle(image,pt1,pt2,(255,0,0),1)
            #pixelcolorBlue = image[x,y,0]
            #pixelcolorGreen = image[x,y,1]
            #pixelcolorRed = image[x,y,2]
            #if pixelcolorBlue in range(blue-5, blue+5) and pixelcolorGreen in range(green-5, green+5) and pixelcolorRed in range(red-5, red+5):
            #    #cv.rectangle(image, (y, x), (y+1, x+1), (255,0,0), 1)
            #    print()
            #    #print("nekaj")
    return bestcoordinates                




framecount = 0
if cap.isOpened() == False:
    print("Ne morem odpret videja")

cv.namedWindow("Kamera")
while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, (280, 340))
    framecount += 1
    if ret == True:
        
        array = moveRectangle(frame)   
        print(array)
        cv.rectangle(frame, (array[1], array[0]), (array[1]+70, array[0]+85), (255,0,0), 1)
       # b = cv.resize(frame, (1080, 1920), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
        cv.imshow("Kamera", frame)
        if cv.waitKey(20) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()