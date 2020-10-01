import cv2
import numpy as np
import imutils
import pytesseract

img = cv2.imread("E:\SemesterBooks\AaveshAndProjects\number_plate.jpg")
img = cv2.resize(img, (620,480))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 13, 15, 15) 

edged = cv2.Canny(gray, 30, 200)
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in contours: 
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print ("No contour detected")
else:
     detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

cv2.imshow('number_plate',new_image)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("programming_fever's License Plate Recognition\n")
print("Detected license plate Number is:",text)
img = cv2.resize(img,(500,300))
Cropped = cv2.resize(Cropped,(400,200))
cv2.imshow('car',img)
cv2.imshow('Cropped',Cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()


#live sketch

# import cv2
# import numpy as np

# # Our sketch generating function
# def sketch(image):
#     # Convert image to grayscale
#     img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Clean up image using Guassian Blur
#     img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    
#     # Extract edges
#     canny_edges = cv2.Canny(img_gray_blur, 30, 65)
    
#     # Do an invert binarize the image 
#     ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
#     return mask


# # Initialize webcam, cap is the object provided by VideoCapture
# # It contains a boolean indicating if it was sucessful (ret)
# # It also contains the images collected from the webcam (frame)
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     cv2.imshow('Our Live Sketcher', sketch(frame))
#     if cv2.waitKey(1) == 27:
#         break
        
# # Release camera and close windows
# cap.release()
# cv2.destroyAllWindows()


#-------------------feature matching--------------

# import numpy as np 
# import cv2
# import matplotlib.pyplot as plt 

# img1=cv2.imread("D:\photos\d3",0)
# img2=cv2.imread("D:\photos\d3",0)

# orb=cv2.ORB_create()

# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)

# bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# matches = bf.match(des1,des2)
# matches = sorted(matches, key = lambda x:x.distance)

# img3 = cv2.drawMatches(img1 ,kp1,img2,kp2,matches[:10],None , flags=2)
# plt.imshow(img3)
# plt.show()

#------------------------harris corner-----------------

# import numpy as np 
# import cv2
# import matplotlib.pyplot as plt

# filename='D:\photos\chess.jpg'
# img = cv2.imread(filename)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# plt.subplot(1,2,1),cv2.imshow('image',img)
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,2,3,0.04)

# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)

# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]

# plt.subplot(1,2,2),cv2.imshow('dst',img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows() 

#-------------------cornerSubPix----------------
# import cv2
# import numpy as np

# filename = 'D:\photos\chess.jpg'
# img = cv2.imread(filename)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# # find Harris corners
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,2,3,0.04)
# dst = cv2.dilate(dst,None)
# ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
# dst = np.uint8(dst)

# # find centroids
# ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# # define the criteria to stop and refine the corners
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

# # Now draw them
# res = np.hstack((centroids,corners))
# res = np.int0(res)
# img[res[:,1],res[:,0]]=[0,0,255]
# img[res[:,3],res[:,2]] = [0,255,0]

# cv2.imshow('subpixel5.png',img)
# if cv2.waitKey(0) & 0xff == 27:
#      cv2.destroyAllWindows() 

#---------------goodFeatureToTrack-------------------------
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

# img = cv2.imread('D:\photos\simple.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
# corners = np.int0(corners)

# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(img,(x,y),2,255,-1)

# plt.imshow(img),plt.show()


#-----------------------sift operation-----

# import numpy as np
# import cv2

# img=cv2.imread("D:\photos\d2.jpg",0)
# sift = cv2.xfeatures2d.SIFT_crate()

# kp = sift.detect(img,None)

# img=cv2.drawKeypoints(img,kp,(0,255,0),flags=0)

# cv2.imshow('image2',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#---------------------surf operation------------------
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

# img = cv2.imread('D:\photos\simple.jpg',0)

# # Initiate FAST object with default values
# fast = cv2.FastFeatureDetector()

# # find and draw the keypoints
# kp = fast.detect(img,None)
# img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))

# # Print all default params
# print("Threshold:"), fast.getInt('threshold')
# print("nonmaxSuppression:") , fast.getBool('nonmaxSuppression')
# print("neighborhood:") , fast.getInt('type')
# print("Total Keypoints with nonmaxSuppression:") , print(len(kp))

# cv2.imshow('fast_true.png',img2)

# # Disable nonmaxSuppression
# fast.setBool('nonmaxSuppression',0)
# kp = fast.detect(img,None)

# print "Total Keypoints without nonmaxSuppression: ", len(kp)

# img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))

# cv2.imshow('fast_false.png',img3)


# ------------------- orb -----------------------------------
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# img = cv2.imread('D:\photos\chess.jpg',0)
#
# # Initiate STAR detector
# orb = cv2.ORB_create()
#
# # find the keypoints with ORB
# kp = orb.detect(img,None)
#
# # compute the descriptors with ORB
# kp, des = orb.compute(img, kp)
#
# # draw only keypoints location,not size and orientation
# img2 = cv2.drawKeypoints(img,kp,(0,255,0), flags=0)
# plt.imshow(img2),plt.show()

#----------------------------------------------------------------