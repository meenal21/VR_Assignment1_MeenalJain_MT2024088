import numpy as np
import cv2
import os
import glob
import imutils

# Get all images from the specified folder.
image_paths = glob.glob('.\Images\Panorama\\*.jpg')
images = []

# Read and show each image.
for image in image_paths:
    img = cv2.imread(image)
    images.append(img)

sift = cv2.SIFT_create()

# Detect and display keypoints on each input image.
for idx, img in enumerate(images):
    kp, des = sift.detectAndCompute(img, None)
    print(f"Image {idx+1} ({image_paths[idx]}): {len(kp)} keypoints detected.")
    img_with_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(".\Images\Panorama\\richkeypoints.png", img_with_kp)
    

# Create a Stitcher object and stitch the images.
imageStitcher = cv2.Stitcher_create()
error, stitched_img = imageStitcher.stitch(images)

if not error:
    # Save and display the raw stitched image.
    cv2.imwrite(".\Panorama\\stitchedOutput.png", stitched_img)

    # Process the stitched image.
    stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(".\Panorama\\thresh_image.png", stitched_img)

    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    minRectangle = mask.copy()
    sub = mask.copy()

    # Erode the mask until we get the minimum rectangle.
    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)

    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)
    cv2.imwrite(".\Panorama\\minRectangle.png", minRectangle)

    x, y, w, h = cv2.boundingRect(areaOI)
    stitched_img = stitched_img[y:y + h, x:x + w]

    cv2.imwrite(".\Panorama\\stitchedOutputProcessed.png", stitched_img)
    cv2.imshow("Stitched_Img",stitched_img)     
    cv2.waitKey(0)

    # Detects and displays keypoints on the processed stitched image.
    kp_final, des_final = sift.detectAndCompute(stitched_img, None)
    print(f"Stitched Processed Image: {len(kp_final)} keypoints detected.")
    stitched_with_kp = cv2.drawKeypoints(stitched_img, kp_final, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(".\Panorama\\Stitched With KP.png", stitched_with_kp)
    cv2.imshow("stitchedOutputProcessed_w.png",stitched_with_kp)
    cv2.waitKey(0)

else:
    print("Images could not be stitched!")
    print("Likely not enough keypoints being detected!")
