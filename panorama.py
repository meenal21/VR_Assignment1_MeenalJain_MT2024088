import cv2 as cv
import numpy as np
import os

def stitch_images(img1, img2):
    """Finds keypoints, matches them, computes homography, and stitches images."""
    sift = cv.SIFT_create()
    key1, desc1 = sift.detectAndCompute(img1, None)
    key2, desc2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    print(f"Found {len(good_matches)} good matches.")  # Debugging

    if len(good_matches) < 10:
        print("Not enough matches to compute homography!")
        return None

    src_pts = np.float32([key1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([key2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

    if H is None:
        print("Homography computation failed!")
        return None

    # Warp img2 onto img1's plane
    height, width = img1.shape[:2]
    result = cv.warpPerspective(img2, H, (width * 2, height))
    result[0:height, 0:width] = img1  # Overlay img1

    return result

# Load images
image_paths = [".\Images\Panorama\\first.jpg", ".\Images\Panorama\\second.jpg", ".\Images\Panorama\\third.jpg"]
images = [cv.imread(path) for path in image_paths]

# Stitch images manually, checking for issues at each step
stitched = stitch_images(images[1], images[0])  # Middle + Right
if stitched is not None:
    stitched = stitch_images(images[2], stitched)  # Left + (Middle + Right)

# Save result
if stitched is not None:
    output_dir = ".\Panorama"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fixed_panorama.png")
    cv.imwrite(output_path, stitched)
    print("Fixed Panorama saved at:", output_path)
else:
    print("Panorama stitching failed!")

