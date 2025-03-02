
# VR Assignment 1

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

![coins.jpeg](https://github.com/meenal21/VR_Assignment_1/blob/main/Images/Coins/coins.jpeg)
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

![coin_1.png](https://github.com/meenal21/VR_Assignment_1/blob/main/Segmented%20Coins/coin_2.png)    ![coin_4.png](https://github.com/meenal21/VR_Assignment_1/blob/main/Segmented%20Coins/coin_4.png)   


## Part 2: Image Stitching :

![Image1](https://github.com/meenal21/VR_Assignment_1/blob/main/Images/Panorama/first.jpg)

![Image2](https://github.com/meenal21/VR_Assignment_1/blob/main/Images/Panorama/second.jpg)

![Image3](https://github.com/meenal21/VR_Assignment_1/blob/main/Images/Panorama/third.jpg)

since SIFT works with only 2 images at a time - so we need to do this activity twice 

#### Import the Image

- The script scans a folder (`.\Images\Panorama\`) to **fetch all images** with a `.jpg` extension.
- The images are then **read using OpenCV** and stored in a list for further processing.
```
image_paths = glob.glob('.\Images\Panorama\\*.jpg')
images = []
for image in image_paths:
    img = cv2.imread(image)
    images.append(img)
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
sift = cv2.SIFT_create()
for idx, img in enumerate(images):
    kp, des = sift.detectAndCompute(img, None)
    print(f"Image {idx+1} ({image_paths[idx]}): {len(kp)} keypoints detected.")
    img_with_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(".\Images\Panorama\\richkeypoints.png", img_with_kp)

```
![RickKeyPoints](https://github.com/meenal21/VR_Assignment_1/blob/main/Panorama/richkeypoints.png)
- `sift.detectAndCompute()` extracts **keypoints** and **descriptors** for each image.
- The **keypoints are drawn** on the images and saved as `richkeypoints.png` for visualization.
- Prints the **number of detected keypoints** for debugging.

Now once this is done, we will match the features and since we have to do this twice, we will create a function to do so:

#### Stitching the Images Together

- `Stitcher_create()` initializes OpenCV's built-in **image stitching algorithm**.
- `stitch(images)` aligns and **blends images seamlessly**.
- If successful, it returns the **stitched panorama** in `stitched_img`

```
imageStitcher = cv2.Stitcher_create()
error, stitched_img = imageStitcher.stitch(images)
```

#### Processing the Stitched Image

- **Thresholding isolates the foreground**, helping in **detecting the stitched region**.
- `findContours()` identifies the **main stitched area**.
- The largest **bounding box** is extracted for further processing.

```
stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite(".\Panorama\\thresh_image.png", stitched_img)

contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
areaOI = max(contours, key=cv2.contourArea)
```

#### Refine the Cropped Panorama

The stitched image is **eroded** to get the **smallest rectangular region** containing useful data.

```
mask = np.zeros(thresh_img.shape, dtype="uint8")
x, y, w, h = cv2.boundingRect(areaOI)
cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

minRectangle = mask.copy()
sub = mask.copy()

while cv2.countNonZero(sub) > 0:
    minRectangle = cv2.erode(minRectangle, None)
    sub = cv2.subtract(minRectangle, thresh_img)
```

This ensures that only the **useful part** of the panorama is kept.

####  Final Cropping and Saving

```
x, y, w, h = cv2.boundingRect(areaOI)
stitched_img = stitched_img[y:y + h, x:x + w]
cv2.imwrite(".\Panorama\\stitchedOutputProcessed.png", stitched_img)
cv2.imshow("Stitched_Img", stitched_img)
cv2.waitKey(0)
```
![Stitched with richkeypoints](https://github.com/meenal21/VR_Assignment_1/blob/main/Panorama/Stitched%20With%20KP.png)
![Stitched Image](https://github.com/meenal21/VR_Assignment_1/blob/main/Panorama/stitchedOutput.png)
