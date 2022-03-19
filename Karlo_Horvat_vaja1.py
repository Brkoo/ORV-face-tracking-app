from turtle import width
import cv2 as cv

print("Hello world")

img = cv.imread("10-15-Day.jpg")

scale_percent = 40

width = int(img.shape[1]* scale_percent /100)
heigth = int(img.shape[0] * scale_percent /100)


dsize = (width, heigth)

img  = cv.resize(img, dsize)

cv.imshow("Image", img)



cv.waitKey(0)



