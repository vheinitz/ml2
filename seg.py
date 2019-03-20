import numpy as np
import cv2
import os, sys
import cv2
import numpy as np
import uuid
from sklearn.externals import joblib
import sklearn.svm as svm
from matplotlib import pyplot as plt

basedir = 'C:\\Development\\PR_DATA\\ModelANA\\img\\'
tmp = os.listdir(basedir)

CellCnt=1

for fn in tmp:

    orig = cv2.imread(basedir+fn)

    img = cv2.pyrDown( orig )
    img = cv2.pyrDown( img )
    cv2.imshow( " orig ", img )
    img = cv2.pyrDown( img )
    img = cv2.pyrDown( img )

    img = cv2.pyrUp( img )
    img = cv2.pyrUp( img )

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
    dtm = dist_transform.max()
    ret, sure_fg = cv2.threshold(dist_transform, 5, 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    #unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    #ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    #markers = markers+1

    # Now, mark the region of unknown with zero
    #markers[unknown==255] = 0

    #markers = cv2.watershed(img,markers)

    #test = np.zeros(img.shape)
    #test[markers == -1] = [255,255,255]
    #img[markers == -1] = [255,0,0]

    dist_transform = np.uint8(dist_transform)

    #image, contours, hierarchy = cv2.findContours(dist_transform, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image, contours, hierarchy = cv2.findContours(sure_fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        al = cv2.arcLength(c,True)
        if al <=0 :
            al=1
        ar = cv2.contourArea(c)
        cr =  (al/(3.1416*2))
        apr= ar/al
        circ = ( apr/cr *2 )
        print "%d %d %f %d %f" % (al, ar, apr, cr, circ)

        if al < 100 and al > 20 and circ > 0.3:
            #cv2.drawContours(img, [c], 0, (255,255,0), 2)
            #M = cv2.moments(c)
            #print M
            x, y, w, h = cv2.boundingRect(c)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            x = (x) *4 -60
            y = (y) *4 -60
            w = 120
            h = 120
            if x > 60 and y>60 and x< 2490 and y < 1850:
                #orig = cv2.rectangle(orig, (x, y), (x + w, y + h), (255,0, 0), 1)
                cell = orig[y:y+120,x:x+120]
                if cell.shape[0] > 0 and CellCnt < 10000:
                    cv2.imwrite( "c:/tmp/cells/%d.png" % CellCnt , cell)
                    CellCnt +=1






    #disconnected = cv2.dilate(dist_transform,kernel,iterations=3)

    cv2.imshow( " image", img )
    #cv2.imshow(" test", test)
    cv2.imshow(" sure_bg", sure_bg)
    cv2.imshow(" opening", opening)
    #cv2.imshow(" orig", orig)


    cv2.waitKey(1)