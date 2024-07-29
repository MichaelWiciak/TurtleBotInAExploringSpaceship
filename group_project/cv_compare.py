import cv2 as cv
 
def compare_images(img1, img2):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    # Apply ratio test
    good = []
    bad = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
        else:
            bad.append([m])

    ratio_good_bad = len(good) / len(bad) if len(bad) != 0 else 0
    
    return ratio_good_bad