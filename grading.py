#Code developed by Yogesh B Kalnar, Scientist,Automation and sensor technology division, ICAR-CIPHET, Ludhiana
# import the necessary packages
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils import contours
from skimage import measure
from imutils import perspective
from scipy.spatial import distance as dist
import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk
##import csv
##header = ['Eccentricity/Elongation’, ‘Area of object: Pixels' ,'ContourPerimeter:', 'Circularity' , 'Centroid(Cx, Cy)', 'Equi_Diameter', 'Width :pixels' , 'Lenght: pixels', 'Width (mm)', 'Length(mm)' , 'Aspect ratio' , 'Solidity’]
##data = [eccentricity, area, perimeter, circularity, cX cY, equi_dia,w, l, dimA_mm, dimB_mm, aspect_ratio, solidity]
##
##with open('countries.csv', 'w', encoding='UTF8', newline='') as f:
##    writer = csv.writer(f)
##
##    # write the header
##    writer.writerow(header)
##
##    # write multiple rows
##    writer.writerows(data)

                          
#Define midpoint of image
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
#############################################################################################
##window = tk.Tk()
##
##window.title("Fruit Adulteration Detection")
##
##window.geometry("500x700")#500x510
##window.configure(background ="gray51")
##
##title = tk.Label(text="Click below to choose picture for testing Adulteration....", background = "lightgreen", fg="Brown", font=("", 17))
##title.grid()  
##dirPath = "grad"
##fileList = os.listdir(dirPath)
##for fileName in fileList:
##    os.remove(dirPath + "/" + fileName)
### C:/Users/sagpa/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
##fileName = askopenfilename(initialdir='D:\\romesh\\codes\\PY\\fruitAdulteration_2_fruit\\dataset\\test', title='Select image for analysis ',
##                       filetypes=[('image files', '.png')])
##dst = "D:\\romesh\\codes\\PY\\fruitAdulteration_2_fruit\\grad"
##print(fileName)
##print (os.path.split(fileName)[-1])
##shutil.copy(fileName, dst)
##load = Image.open(fileName)
##render = ImageTk.PhotoImage(load)
##img = tk.Label(image=render, height="400", width="500")
##img.image = render
##img.place(x=0, y=0)
##img.grid(column=0, row=1, padx=10, pady = 10)


#############################################################################################

# load the image, convert it to grayscale, blur it slightly, and threshold it
Original_img = cv2.imread("D:\\romesh\\codes\\PY\\fruitAdulteration_2_fruit\\dataset\\test\\.png")
scale_percent = 50 # percent of original size
width = int(Original_img.shape[1] * scale_percent / 100)
height = int(Original_img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
image = cv2.resize(Original_img, dim, interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 1)
thresh= cv2.threshold(blurred, 142, 255, cv2.THRESH_BINARY)[1]


##cv2.imshow("Threshold Image", thresh)
# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
#Grab the contours: cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None
# loop over the contours
for c in cnts:
        
###########################################################################################
# if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 1500:
            continue
        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (255, 20, 255), 2)
        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (255, 0, 0), -1)
        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the bottom-right and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                (0, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                (0, 255, 0), 2)
        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
                pixelsPerMetric = dA / 0.86142# 1=dia of coin/object left most in inch
            # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        dimA_mm = dimA*25.4
        dimB_mm = dimB*25.4
        # draw the object sizes on the image
        cv2.putText(orig, "{:.2f}mm".format(dimA_mm),
                (int(tltrX  +15), int(tltrY +10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (25, 25, 255), 2)
        cv2.putText(orig, "{:.2f}mm".format(dimB_mm),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (1, 255, 1), 2)

        ############################################################################################
        # compute the shape moments
        M = cv2.moments(c)
        #print('Moments', M)
        #Elongation/Eccemtricity of the shape based on moments
        a1 = (M['mu20'] + M['mu02']) / 2
        a2 = np.sqrt(4 * M['mu11'] ** 2 + (M['mu20'] - M['mu02']) ** 2) / 2
        eccentricity = np.sqrt(1 - (a1 - a2) / (a1 + a2))
        print('Eccentricity/Elongation:', eccentricity)
        #Area of the contour
        area = cv2.contourArea(c)
        print ('Area of object, in Pixels:', area)
        #Perimeter of contour
        perimeter = cv2.arcLength(c,True)
        print ('Contour Perimeter:', perimeter)
        #Circularity: C = 4_pi_A/P^2 ~ 12.57_A/P^2, where C:circularity, A:area and P:perimeter.
        circularity = 4*np.pi*(area/(perimeter*perimeter))
        print('Circularity:', circularity)
        #centroid: (cX, cY)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print('Centroid(Cx, Cy):', cX,cY)
        #Equivalent dia of shape
        equi_dia = np.sqrt(4*area/np.pi)
        print ('Equi_Diameter:', equi_dia)
        #Aspect ratio=ratio of width to height of bounding rect of the object
        x,y,w,l = cv2.boundingRect(c)
        aspect_ratio = float(w)/l
        print('Width in pixels):',w)
        print('Lenght in  pixels):',l)
        print ('Width (mm):', dimA_mm)
        print ('Length(mm):', dimB_mm)
        print('Aspect ratio (width to height of bounding rect):',aspect_ratio)
        #Solidity: the ratio of contour area to its convex hull area
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area
        print('Solidity:',solidity)
##        cv2.imshow("Orig", orig)
        # draw the contour and center of the shape on the image
        cv2.drawContours(image, [c], -1, (55, 255, 155), 3)
        cv2.circle(image, (cX, cY), 7, (255, 55, 255), -1)
        cv2.putText(image, "Center", (cX - 20, cY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # show the image
        #cv2.imshow("Image with contour", image)
        cv2.waitKey(0)
        print('======next shape=======')
	
#For subplot of 6 images 3rx2c
titles = ['Input image']
images = [image]
for i in range(1):
    plt.subplot(1,1,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.show()
