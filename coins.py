
import cv2 as cv
import numpy as np
import os

path = "F:\Downloads\IIITB CourseWork\Sem 2\VR\Assignment1\coins.jpeg"
path =  "F:\Downloads\IIITB CourseWork\Sem 2\VR\Assignment1\coins_1.jpeg"
#path =  "F:\Downloads\IIITB CourseWork\Sem 2\VR\Assignment1\coins_2.jpeg"
path = ".\Images\coins.jpeg"
segment_path = ".\Segmented Coins"

os.makedirs(segment_path, exist_ok=True)

#Read the image and save in img
img = cv.imread(path)

#Since the image is very large, we will reduce the size to half
half = cv.resize(img, (0, 0), fx=0.5, fy=0.5)

#Convert the BGR to Grayscale for Gaussian Blurring, required for Canny Edge Detection
gray = cv.cvtColor(half, cv.COLOR_BGR2GRAY)

#Using Kernel size as 9 - 
blur = cv.GaussianBlur(gray, (9,9),0)

#Since the image has soft shadow which cannot be resolved by global thresholding, we use adaptive thresholding -
adapt = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2) 

#We blur the image again - once the shadows are removed - 
blur = cv.GaussianBlur(adapt, (9,9),0)

#Now that the image is appropriately smoothened, we use Canny Edge Detection to detect the edges in the image - 
canny = cv.Canny(blur,180,250)

#but this still has a lot of sharp edges, broken edges - which we dilate - due to texture which is not necessary for proper edges
# we use a kernel of odd numbers - here 3x3 kernel
kernel = np.ones((3,3), np.uint8)
dilate = cv.dilate(canny,kernel, iterations=1)

filename = os.path.join(segment_path,f"coin_edge_detection.png")
cv.imwrite(filename, dilate)

#After dilating, we need to find only the outer edges using findContours - 
contours,_ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#valid_contours = np.zeros_like(dilate)
#circular_contours = np.zeros_like(dilate)

#initialising the coin_count to 0
coin_count = 0

#Path to store the segmented coin photos


valid_conts = []
for cnt in contours:
	area = cv.contourArea(cnt) 
	perimeter = cv.arcLength(cnt, True) 
	if perimeter == 0: 
		continue 
	
	circularity = (4 * np.pi * area) / (perimeter ** 2)
	
	if circularity > 0.7 and cv.contourArea(cnt) >  500:
		#cv.drawContours(valid_contours, [cnt], -1, (255),thickness=2)
		cv.drawContours(half, [cnt], -1, (0,255,0),thickness=2)
		(x, y), radius = cv.minEnclosingCircle(cnt) 
		center = (int(x), int(y)) 
		radius = int(radius)
		valid_conts.append(cnt)
		x1, y1 = max(0, center[0] - radius), max(0, center[1] - radius) 
		x2, y2 = min(half.shape[1], center[0] + radius), min(half.shape[0], center[1] + radius) 
		# Crop the coin 
		coin = half[y1:y2, x1:x2]
		
		filename = os.path.join(segment_path,f"coin_{coin_count}.png")
		cv.imwrite(filename, coin) 
		coin_count += 1
	
		#cv.circle(circular_contours, center, radius, (255), thickness=2)
		cv.circle(half, center, radius, (255), thickness=2)

filename = os.path.join(segment_path,f"coin_segment_all.png")
cv.imwrite(filename, half) 

print(f"The coin count is {coin_count}")