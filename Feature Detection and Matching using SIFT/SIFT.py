

import cv2

from matplotlib import  pyplot as plt

DataSet=[]
Queryimg=cv2.imread("query.jpg")

DataSet=["1.jpg","2.jpg","3.jpg","4.jpg","5.jpg","6.jpg","7.jpg","8.jpg"]

sift=cv2.xfeatures2d.SIFT_create()

Qkp,Qdes=sift.detectAndCompute(Queryimg,None)
bf=cv2.BFMatcher()
Descriptors=[]
KeyPoints=[]
Matches=[]
Distances=[]

for i in range(len(DataSet)):
    img=cv2.imread(DataSet[i])
    kp,des=sift.detectAndCompute(img,None)
    M=bf.match(Qdes,des)
    dist=0

    for j in range(len(M)):
        dist+=M[i].distance
    if (i==0):
        MinDist=dist
        MinIdx=i
    elif (dist<MinDist):
        MinDist=dist
        MinIdx=i
    Descriptors.append(des)
    KeyPoints.append(kp)
    Distances.append(dist)
    Matches.append(M)



BestMatch=cv2.imread(DataSet[MinIdx])
cv2.DRAW_MATCHES_FLAGS_DEFAULT
img2=cv2.drawMatches(Queryimg,Qkp,BestMatch,KeyPoints[MinIdx],Matches[MinIdx][:100],flags=2,outImg=None)


# plt.imshow(img2)
# plt.show()

cv2.imshow("Result",img2)
cv2.waitKey()
