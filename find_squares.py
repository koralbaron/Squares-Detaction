import cv2
import numpy as np
import argparse

def fillterImage(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshImg = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)
    kernel = np.ones((2, 2), np.uint8)
    threshImg = cv2.erode(threshImg, kernel, iterations=2)
    threshImg = cv2.dilate(threshImg, kernel, iterations=1)
    cv2.bitwise_not(threshImg, threshImg)
    return threshImg


# draw squares mark in straight lines
def markSquaresPretty(img, x, y, w, h, color, thickness):
    a = [[x, y], [x + w, y], [x + w, h + y], [x, y + h]]
    cv2.drawContours(img, [np.array(a)], 0, color, thickness)


# check if the shape is a square and if so returns true.
def isSquare(approx, ar, minWidth, minWidthThresh, threshold):
    if 0.93 <= ar <= 1.07 and minWidth > minWidthThresh:
        if abs(approx[0][0][0] - approx[3][0][0]) < threshold and abs(approx[1][0][0] - approx[2][0][0]) < threshold and \
                abs(approx[0][0][1] - approx[1][0][1]) < threshold and abs(approx[3][0][1] - approx[2][0][1]) < threshold:
            return True
    return False


# find and mark all the squares in the image
def main(args):

    args.color = tuple([int(x) for x in args.color])
    img = cv2.imread(args.img_path)
    threshImg = fillterImage(img)
    cv2.imshow('Black and white image', threshImg)
    contours, _ = cv2.findContours(threshImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cont in contours:
        approx = cv2.approxPolyDP(cont, 0.07 * cv2.arcLength(cont, True), True)
        if len(approx) == 4:  # if shape is quadrangle
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if isSquare(approx, ar, w, 4, 3):
                markSquaresPretty(img, x, y, w, h, args.color, args.thickness)
    cv2.imwrite("markedImg.jpg", img)
    cv2.imshow('marked image', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some args')
    parser.add_argument("--img_path", default="image001.jpg")
    parser.add_argument("--color", default=[0, 0, 255],  nargs=3, help="color of the marks, insert in this format: --color 0 255 0")
    parser.add_argument("--thickness", default=2, type=int, help="thickness of the marks")
    args = parser.parse_args()
    main(args)
