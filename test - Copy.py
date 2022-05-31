import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk
import imutils
from imutils import contours
from skimage import measure
from imutils import perspective
from scipy.spatial import distance as dist
import cv2
import numpy as np
##import tqdm

from tqdm import tqdm
window = tk.Tk()

window.title("Fruit Adulteration Detection")

window.geometry("500x700")#500x510
window.configure(background ="gray51")

title = tk.Label(text="Click below to choose picture for testing Adulteration....", background = "lightgreen", fg="Brown", font=("", 17))
title.grid()   
def grading():
    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    verify_dir = 'testpicture'
    
        #Define midpoint of image
    file=''
    for img in tqdm(os.listdir(verify_dir)):
        file = img
# load the image, convert it to grayscale, blur it slightly, and threshold it
    Original_img = cv2.imread("C:\\Users\\DELL\\Desktop\\fruitAdulteration_2_fruit\\testpicture\\"+file)
    scale_percent = 50 # percent of original size
    width = int(Original_img.shape[1] * scale_percent / 100)
    height = int(Original_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    image = cv2.resize(Original_img, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1)
    thresh= cv2.threshold(blurred, 142, 255, cv2.THRESH_BINARY)[1]


    
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
      
        #Area of the contour
        area = cv2.contourArea(c)
        print ('Area of object, in Pixels:', area)
        if area<= 6000:
            print("Grade A")
        elif area> 6000:
            print("Grade B")
        #Perimeter of contour
        perimeter = cv2.arcLength(c,True)
        print ('Contour Perimeter:', perimeter)
        #Circularity: C = 4_pi_A/P^2 ~ 12.57_A/P^2, where C:circularity, A:area and P:perimeter.
        circularity = 4*np.pi*(area/(perimeter*perimeter))
        print('Circularity:', circularity)
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
 
        print('======next shape=======')
            
    #For subplot of 6 images 3rx2c
    titles = ['Input image']
    images = [image]


def analysis():
    def BlackRotCankerApple():
        window.destroy()
        window1 = tk.Tk()

        window1.title("Fruit Adulteration Detection")

        window1.geometry("500x510")
        window1.configure(background="gray51")

        def exit():
            window1.destroy()
        rem = "The remedies for BlackRotCanker Spot are:\n\n "
        remedies = tk.Label(text=rem, background="lightgreen",
                          fg="Brown", font=("", 15))
        remedies.grid(column=0, row=7, padx=10, pady=10)
        rem1 = " remove mummified fruit and sanitize with Thiophanate-Methyl spray"
        remedies1 = tk.Label(text=rem1, background="lightgreen",
                            fg="Black", font=("", 12))
        remedies1.grid(column=0, row=8, padx=10, pady=10)

        rem2 = "black rot have 37% of pottasium added in fruit"
        remedies2 = tk.Label(text=rem2, background="lightgreen",
                            fg="Black", font=("", 12))
        remedies2.grid(column=0, row=9, padx=10, pady=10)

        button = tk.Button(text="Exit", command=exit)
        button.grid(column=0, row=10, padx=20, pady=20)

        window1.mainloop()

    def BrownRotApple():
        window.destroy()
        window1 = tk.Tk()

        window1.title("Fruit Adulteration Detection")

        window1.geometry("650x510")
        window1.configure(background="lightgreen")

        def exit():
            window1.destroy()
        rem = "The remedies for BrownRot are: "
        remedies = tk.Label(text=rem, background="lightgreen",
                          fg="Brown", font=("", 15))
        remedies.grid(column=0, row=7, padx=10, pady=10)
        rem1 = " Thiophanate-Methyl spray,Organic Spray,Captan sanitizing spray"
        remedies1 = tk.Label(text=rem1, background="lightgreen",
                             fg="Black", font=("", 12))
        remedies1.grid(column=0, row=8, padx=10, pady=10)

        button = tk.Button(text="Exit", command=exit)
        button.grid(column=0, row=9, padx=20, pady=20)

        window1.mainloop()
    def ScabApple():
        window.destroy()
        window1 = tk.Tk()

        window1.title("Fruit Adulteration Detection")

        window1.geometry("650x510")
        window1.configure(background="lightgreen")

        def exit():
            window1.destroy()
        rem = "The remedies for scab are: "
        remedies = tk.Label(text=rem, background="lightgreen",
                          fg="Brown", font=("", 15))
        remedies.grid(column=0, row=7, padx=10, pady=10)
        rem1 = "cooper- and sulphur- based fungicides"
        remedies1 = tk.Label(text=rem1, background="lightgreen",
                             fg="Black", font=("", 12))
        remedies1.grid(column=0, row=8, padx=10, pady=10)

        button = tk.Button(text="Exit", command=exit)
        button.grid(column=0, row=9, padx=20, pady=20)

        window1.mainloop()


    def FungalOranges():
        window.destroy()
        window1 = tk.Tk()

        window1.title("Fruit Adulteration Detection")

        window1.geometry("650x510")
        window1.configure(background="lightgreen")

        def exit():
            window1.destroy()
        rem = "The remedies for Fungal are: "
        remedies = tk.Label(text=rem, background="lightgreen",
                          fg="Brown", font=("", 15))
        remedies.grid(column=0, row=7, padx=10, pady=10)
        rem1 = " Horticultural Oils,dormant spray"
        remedies1 = tk.Label(text=rem1, background="lightgreen",
                             fg="Black", font=("", 12))
        remedies1.grid(column=0, row=8, padx=10, pady=10)

        button = tk.Button(text="Exit", command=exit)
        button.grid(column=0, row=9, padx=20, pady=20)

        window1.mainloop()
    def MelanoseOranges():
        window.destroy()
        window1 = tk.Tk()

        window1.title("Fruit Adulteration Detection")

        window1.geometry("650x510")
        window1.configure(background="lightgreen")

        def exit():
            window1.destroy()
        rem = "The remedies for Melanose are: "
        remedies = tk.Label(text=rem, background="lightgreen",
                          fg="Brown", font=("", 15))
        remedies.grid(column=0, row=7, padx=10, pady=10)
        rem1 = " Searles Copper Oxychloride"
        remedies1 = tk.Label(text=rem1, background="lightgreen",
                             fg="Black", font=("", 12))
        remedies1.grid(column=0, row=8, padx=10, pady=10)

        button = tk.Button(text="Exit", command=exit)
        button.grid(column=0, row=9, padx=20, pady=20)

        window1.mainloop()

    def PencililliumDigitatumOranges():
        window.destroy()
        window1 = tk.Tk()

        window1.title("Fruit Adulteration Detection")

        window1.geometry("650x510")
        window1.configure(background="lightgreen")

        def exit():
            window1.destroy()
        rem = "The remedies for PencililliumDigitatum are: "
        remedies = tk.Label(text=rem, background="lightgreen",
                          fg="Brown", font=("", 15))
        remedies.grid(column=0, row=7, padx=10, pady=10)
        rem1 = "Orchard and packinghouse sanitation is required"
        remedies1 = tk.Label(text=rem1, background="lightgreen",
                             fg="Black", font=("", 12))
        remedies1.grid(column=0, row=8, padx=10, pady=10)

        button = tk.Button(text="Exit", command=exit)
        button.grid(column=0, row=9, padx=20, pady=20)

        window1.mainloop()

    def PencililliumMoldOranges():
        window.destroy()
        window1 = tk.Tk()

        window1.title("Fruit Adulteration Detection")

        window1.geometry("520x510")
        window1.configure(background="lightgreen")

        def exit():
            window1.destroy()
        rem = "The remedies for PencililliumMold  are: "
        remedies = tk.Label(text=rem, background="lightgreen",
                          fg="Brown", font=("", 15))
        remedies.grid(column=0, row=7, padx=10, pady=10)

        rem1 = " postharvest PYR treatment"
        remedies1 = tk.Label(text=rem1, background="lightgreen",
                             fg="Black", font=("", 12))
        remedies1.grid(column=0, row=8, padx=10, pady=10)

        button = tk.Button(text="Exit", command=exit)
        button.grid(column=0, row=9, padx=20, pady=20)

        window1.mainloop()


    ##########################################################################################

    
    import cv2  # working with, mainly resizing, images
    import numpy as np  # dealing with arrays
    import os  # dealing with directories
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import \
        tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
    
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'FruitAdulteration-{}-{}.model'.format(LR, '2conv-basic')   ##FruitAdulteration-0.001-2conv-basic.model
    verify_dir = 'testpicture'
    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
########    from tensorflow.python.framework import ops
########    ops.reset_default_graph()
##    tf.reset_default_graph()
    tf.compat.v1.reset_default_graph()
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 9, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        model_out = model.predict([data])[0]
        print(model_out)
        print('model {}'.format(np.argmax(model_out)))

        if np.argmax(model_out) == 0:
            str_label = 'BlackRotCankerApple'
        elif np.argmax(model_out) == 1:
            str_label = 'BrownRotApple'
        elif np.argmax(model_out) == 2:
            str_label = 'ScabApple'
        elif np.argmax(model_out) == 3:
            str_label = 'FungalOranges'
        elif np.argmax(model_out) == 4:
            str_label = 'MelanoseOranges'
        elif np.argmax(model_out) == 5:
            str_label = 'PencililliumDigitatumOranges'
        elif np.argmax(model_out) == 6:
            str_label = 'PencililliumMoldOranges'
        elif np.argmax(model_out) == 7:
            str_label = 'FreshApples'
        elif np.argmax(model_out) == 8:
            str_label = 'FreshOranges'

        status=''
        if str_label == 'BlackRotCankerApple':
            Adulterationname = "BlackRotCankerApple  "
            Adulteration = tk.Label(text='Fruit name : ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='BlackRotCankerApple', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=BlackRotCankerApple)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)
        elif str_label == 'BrownRotApple':
            Adulterationname = "BrownRotApple"
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='BrownRotApple', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=BrownRotApple)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)
        elif str_label == 'ScabApple':
            Adulterationname = "ScabApple "
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='ScabApple', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=ScabApple)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)


        elif str_label == 'FungalOranges':
            Adulterationname = "FungalOranges  "
            Adulteration = tk.Label(text='Fruit name : ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='FungalOranges', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=FungalOranges)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)
        elif str_label == 'MelanoseOranges':
            Adulterationname = "MelanoseOranges"
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='MelanoseOranges', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=MelanoseOranges)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)
        elif str_label == 'PencililliumDigitatumOranges':
            Adulterationname = "PencililliumDigitatumOranges "
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='PencililliumDigitatumOranges', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=PencililliumDigitatumOranges)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)

        elif str_label == 'PencililliumMoldOranges':
            Adulterationname = "PencililliumMoldOranges "
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='PencililliumMoldOranges', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=PencililliumMoldOranges)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)


        elif str_label =='FreshOranges' :
            status= 'Healthy' + str_label
            message = tk.Label(text='Status: '+status, background="gray51",
                           fg="Brown", font=("", 15))
            message.grid(column=1, row=4, padx=15, pady=15)
            r = tk.Label(text='ORANGE is healthy', background="lightgreen", fg="Black",font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            r1 = tk.Label(text='formaldehyde is low,this orange is good to eat', background="lightgreen", fg="Black",font=("", 12))
            r1.grid(column=0, row=6, padx=10, pady=10)
            print("formaldehyde is low,this orange is good to eat")
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=9, padx=20, pady=20)


##        message = tk.Label(text='Status: '+status, background="lightgreen",
##                           fg="Brown", font=("", 15))
##        message.grid(column=0, row=3, padx=10, pady=10)

        elif str_label == 'FreshApples':
            status= 'Healthy' + str_label
            message = tk.Label(text='Status: '+status, background="gray51",
                           fg="Brown", font=("", 15))
            message.grid(column=3, row=5, padx=15, pady=15)
            r = tk.Label(text='APPLE is healthy', background="lightgreen", fg="Black",font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            r1 = tk.Label(text='APPLE is healthy', background="lightgreen", fg="Black",font=("", 15))
            r1.grid(column=0, row=6, padx=10, pady=10)
            print("formaldehyde is low,this  Apple is good to eat")
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=9, padx=20, pady=20)


        message = tk.Label(text='Status: '+status, background="lightgreen",
                           fg="Brown", font=("", 15))
        message.grid(column=0, row=3, padx=10, pady=10)

def openphoto():
    dirPath = "testpicture"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
    # C:/Users/sagpa/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
    fileName = askopenfilename(initialdir='C:\\Users\\DELL\\Desktop\\fruitAdulteration_2_fruit\\dataset\\test', title='Select image for analysis ',
                           filetypes=[('image files', '.png')])
    dst = "C:\\Users\\DELL\\Desktop\\fruitAdulteration_2_fruit\\testpicture"
    print(fileName)
    print (os.path.split(fileName)[-1])
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="400", width="500")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady = 10)
    title.destroy()
    button1.destroy()
    button2 = tk.Button(text="Analyse Image", command=analysis)
    button2.grid(column=0, row=3, padx=10, pady = 10)
    buttong = tk.Button(text="Grade Image", command=grading)
    buttong.grid(column=0, row=2, padx=10, pady = 10)
    
button1 = tk.Button(text="Get Photo", command = openphoto)
button1.grid(column=0, row=1, padx=10, pady = 10)



window.mainloop()



