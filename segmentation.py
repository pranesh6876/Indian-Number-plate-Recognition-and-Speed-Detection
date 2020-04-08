import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure



originalImage = cv2.imread('d2.jpg')
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 90, 255, cv2.THRESH_BINARY)
blackAndWhiteImage=np.array(~blackAndWhiteImage)
#print(blackAndWhiteImage)
labelled_plate = measure.label(blackAndWhiteImage)
fig, ax1 = plt.subplots(1)
ax1.imshow(blackAndWhiteImage, cmap="gray")
#cv2.imshow('Black white image', blackAndWhiteImage)
#cv2.imshow('Original image',originalImage)
print(blackAndWhiteImage.shape)
#cv2.imshow('Gray image', grayImage)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


from skimage.transform import resize
from skimage.measure import regionprops
import matplotlib.patches as patches
character_dimensions = (0.35*blackAndWhiteImage.shape[0], 0.60*blackAndWhiteImage.shape[0], 0.05*bla
min_height, max_height, min_width, max_width = character_dimensions
print(character_dimensions)


characters = []
counter=0
column_list = []
f=[]
for regions in regionprops(labelled_plate):
y0, x0, y1, x1 = regions.bbox
#print(regions.bbox)
f.append(regions.bbox)
print(f)


print(f[0])
y0,x0,y1,x1=f[0]
print(y0,x0,y1,x1)
e=[]
for i in range(0 7):
y0,x0,y1,x1=f[i]
print(y0,x0,y1,x1)
a=blackAndWhiteImage[y0:y1:,x0:x1]
e.append(a)
#fig, ax1 = plt.subplots(3,3,i)
#ax1.imshow(a, cmap="gray")
#plt.imshow(e[i],cmap="gray")
plt.imshow(e[0],cmap="gray")