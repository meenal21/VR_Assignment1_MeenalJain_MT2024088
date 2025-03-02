# Visual Recognition - Assignment 1

**Part 1 : Coin Detection**
a. Detect all coins in the image (2 Marks)
- Use edge detection, to detect all coins in the image. 
- Visualize the detected coins by outlining them in the image. 

b. Segmentation of Each Coin (3 Marks) 
- Apply region-based segmentation techniques to isolate individual coins from the image.
- Provide segmented outputs for each detected coin. 

c. Count the Total Number of Coins (2 Marks)
- Write a function to count the total number of coins detected in the image. 
- Display the final count as an output.

**Part 2: Create a stitched panorama from multiple overlapping images.** ___________________________________________________________________ 

a. Extract Key Points (1 Mark)
- Detect key points in overlapping images. 

b. Image Stitching (2 Marks)
- Use the extracted key points to align and stitch the images into a single panorama. 
- Provide the final panorama image as output.

### Requirements:

Install Anaconda from its website

Use Anaconda Prompt

Create virtual environment using 
```
conda create --name open_cv python=3.9
```

Activate the virtual environment:
```
conda activate open_cv
```

Install OpenCV
```
conda install -c conda-forge opencv
```

-c conda-forge -> tells that fetch opencv from conda-forge instead of Anaconda channel

```
import cv2 print(cv2.__version__)
```


##### Apart from this install these libraries

`os`
`cv2`
`numpy`

### Part 1: Coin Detection

![[coins.jpeg]]
```
import cv2 as cv
path = "F:\Downloads\IIITB CourseWork\Sem 2\VR\Assignment1\coins.jpeg"
path =  "F:\Downloads\IIITB CourseWork\Sem 2\VR\Assignment1\coins_1.jpeg"
#path =  "F:\Downloads\IIITB CourseWork\Sem 2\VR\Assignment1\coins_2.jpeg"
img = cv.imread(path)
```

```
cv.imshow("Image",img)
cv.waitKey(0)
cv.destroyAllWindows()
```

```
half = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
cv.imshow("Half", half)
cv.waitKey(0)
cv.destroyAllWindows()
```

convert the image to grayscale as we only need to do edge detection - lesser number of channels to deal with
```
gray = cv.cvtColor(half, cv.COLOR_BGR2GRAY)
cv.imshow("Gray",gray)
cv.waitKey(0)
cv.destroyAllWindows()
```

###### Canny Edge Detector:
	- Gaussian Blur
	- Apply Gradients - Sobel Filters
	- Non Maximum Suppression - to thin out edges
	- 

Blur using Gaussian Blur - for smoother edge detection, is there are sharp edges, then small texture details will also be picked up - so only short meaningful edges to be picked up. Also this removes noise - which helps the Canny Edge detector

###### Gaussian Blur -
`cv.GaussianBlur(src, ksize, sigMax)`

`ksize`: kernel size - we take in odd numbers - (3,3),(5,5)
`signMax`: Standard Deviation - 0 lets CV decide

smaller number -> slighter blur ; larger number -> stronger blur

```
blur = cv.GaussianBlur(gray, (9,9),0)
cv.imshow("Blur",blur)
cv.waitKey(0)
cv.destroyAllWindows()
```

```
blur = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2) 
cv.imshow("Adaptive Threshold", blur) 
cv.waitKey(0) 
cv.destroyAllWindows()
```

```
blur = cv.GaussianBlur(blur, (9,9),0)
cv.imshow("Blur",blur)
cv.waitKey(0)
cv.destroyAllWindows()
```

Now apply Canny Edge Detector:
`cv.Canny(image, threshold1, threshold2)`

`image`: input grayscale image
`threshold1`: lower - weaker edges will be discarded by this
`threshold2`: upper - stronger edges are kept

```
canny = cv.Canny(blur,180,250)
cv.imshow("Canny",canny)
cv.waitKey(0)
cv.destroyAllWindows()
```


I see a lot of broken edges - use Dilation:
- helps connect broken edges by expanding the white pixels - fixes gaps by connecting close edges
- use a kernel - small kernel of 3 x 3

Create a kernel - `kernel = np.ones((3, 3), np.uint8)`

Dilate: `cv.dilate(canny,kernel, iterations=1)
```
import numpy as np

kernel = np.ones((3,3), np.uint8)
dilate = cv.dilate(canny,kernel, iterations=1)
cv.imshow("Dilate",dilate)
cv.waitKey(0)
cv.destroyAllWindows()
```

now I see enough dilation, outside, I would like to find the contours -

`cv.findContours` : given the edge (binary) image - it extracts the contours - list of boundaries and the hierarchy - inner outer relationship

`cv.findContours(image, mode, method)`

`mode`: Retrieval mode - 
	RETR_EXTERNAL - only the outer edges 
	RETR_LIST - no hierarchy - shows all
	RETR_CCOMP - two level org
	RETR_TREE - relationship tree

`method`: Approximation method:
	CHAIN_APPROX_NONE - saves all the points
	CHAIN_APPROX_SIMPLE - removes redundant pts

To filter out the necessary contours - we use 
`contourArea` -> reduces small/noise contours

```
import os
contours,_ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

valid_contours = np.zeros_like(dilate)
circular_contours = np.zeros_like(dilate)
coin_count = 0
segment_path = ".\Segmented Coins"
valid_conts = []
for cnt in contours:
	area = cv.contourArea(cnt) 
	perimeter = cv.arcLength(cnt, True) 
	if perimeter == 0: 
		continue 
	
	circularity = (4 * np.pi * area) / (perimeter ** 2)
	
	if circularity > 0.7 and cv.contourArea(cnt) >  500:
		cv.drawContours(valid_contours, [cnt], -1, (255),thickness=2)
		cv.drawContours(half, [cnt], -1, (0,255,0),thickness=2)
		(x, y), radius = cv.minEnclosingCircle(cnt) 
		center = (int(x), int(y)) 
		radius = int(radius)
		valid_conts.append(cnt)
		x1, y1 = max(0, center[0] - radius), max(0, center[1] - radius) 
		x2, y2 = min(half.shape[1], center[0] + radius), min(half.shape[0], center[1] + radius) 
		# Crop the coin 
		coin = half[y1:y2, x1:x2]
		os.makedirs(segment_path, exist_ok=True)
		filename = os.path.join(segment_path,f"coin_{coin_count}.png")
		cv.imwrite(filename, coin) 
		coin_count += 1
	
		cv.circle(circular_contours, center, radius, (255), thickness=2)
		cv.circle(half, center, radius, (255), thickness=2)

cv.imshow("Valid_Contours",valid_contours)
cv.waitKey(0)
cv.imshow("Circular_Contours",circular_contours)
cv.waitKey(0)
cv.destroyAllWindows()

cv.drawContours(img,valid_conts, -1,(0,255,0), 2)
cv.imshow("Drawn Image", half)
cv.waitKey(0)
cv.destroyAllWindows()


cv.imshow("Drawn Image", half)
cv.waitKey(0)
cv.destroyAllWindows()
```

Displaying the Coins - Segmented Coins 
```
print(f`Total number of coins - {coin_count}')
```

Now I want to draw the contours in a circular fashion - 

**Check Circularity** using the formula:

- Use Min Enclosing Circle (`cv2.minEnclosingCircle()`)

The visualisation of few of the coins is 

![[coin_1.png]]    ![[coin_5.png]]   


## Part 2: Image Stitching :

since SIFT works with only 2 images at a time - so we need to do this activity twice 

#### Import the Image
```
import cv2 as cv
path1 = ".\Images\\first.jpg"

path2 = ".\Images\\second.jpg"

path3 = ".\Images\\third.jpg"

img1 = cv.imread(path1)
img2 = cv.imread(path2)
img3 = cv.imread(path3)
```

#### Convert from BGR to GrayScale

for keeping only 1 channel - 
```
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) 
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) 
gray3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY) 
```

#### For Feature Detection, use SIFT - 

Initialise and detect keypoints and compute descriptors
```
sift = cv.SIFT_create()

key1, desc1 = sift.detectAndCompute(gray1, None)
key2, desc2 = sift.detectAndCompute(gray2, None)
key3, desc3 = sift.detectAndCompute(gray3, None)
```


Now once this is done, we will match the features and since we have to do this twice, we will create a function to do so:

#### Match the Features

We use FLANN - as it helps find the best matches between the key points from the key point -
```
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) 
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params) 
matches = flann.knnMatch(desc1, desc2, k=2)
```

We collect the good matches based on the threshold of 0.7 - Lowe's Ratio Test:
```
good_matches = [] 

for m, n in matches: 
	if m.distance < 0.75 * n.distance: 
		good_matches.append(m)
```

#### Extracting Matched Keypoints and computing Homography:
```
import numpy as np
src_pts = np.float32([key1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([key2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
```

Calculate the transformation matrix (Homography) between the images:
```
H,_ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
```

H - 3x3 matrix which aligns image1 onto image2
**RANSAC** - Random Sample Consensus removes outliers to improve accuracy
5.0 - maximum projection error allowed

#### Warp image1 onto image2 - Stitching the images
```
height, width, _ = img2.shape
warped_img1 = cv.warpPerspective(img1, H, (width * 2, height))

panaroma_path = ".\Panaroma"
os.makedirs(panaroma_path, exist_ok=True)
filename = os.path.join(panaroma_path,f"panaroma.png")
cv.imwrite(filename, warped_img1)
```


```
def match_and_warp(img1, img2, keypoints1, descriptors1, keypoints2, descriptors2):
    # Use FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    # Apply Lowe's ratio test
    
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
        
    
# Stitch first two images
stitched12 = match_and_warp(img2, img3, key2, desc2, key3, desc3)
```

```
panorama_path = ".\Panorama"
os.makedirs(panorama_path, exist_ok=True)
filename = os.path.join(panorama_path,f"first_pan.png")
cv.imwrite(filename, stitched12)
```

```
# Convert stitched12 to grayscale for next match
gray12 = cv.cvtColor(stitched12, cv.COLOR_BGR2GRAY)

# Detect keypoints in stitched12 and image3
keypoints12, descriptors12 = sift.detectAndCompute(gray12, None)

# Stitch the third image
final_panorama = match_and_warp(img1, stitched12, key1, desc1, keypoints12, descriptors12)

```

```
panorama_path = ".\Panorama"
os.makedirs(panorama_path, exist_ok=True)
filename = os.path.join(panorama_path,f"panorama.png")
cv.imwrite(filename, final_panorama)
```
