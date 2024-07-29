# from group_project import AJBastroalign as aa
import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_image(image, name="image", wait=True):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.resizeWindow(name, 320, 240)
    if wait:
        cv2.waitKey(0)
    return


def astro_stitch(earth, moon):
    sift = cv2.SIFT_create()

    # will need to swap these if the images are in the wrong order
    r_image = moon
    l_image = earth

    l_keypoints, descriptors1 = sift.detectAndCompute(l_image, None)
    r_keypoints, descriptors2 = sift.detectAndCompute(r_image, None)

    l_draw = cv2.drawKeypoints(earth, l_keypoints, None)
    r_draw = cv2.drawKeypoints(moon, r_keypoints, None)
    # show_image(l_draw, "l_draw", wait=False)
    # show_image(r_draw, "r_draw", wait=False)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match the descriptors using brute-force matching
    matches_bf = bf.match(descriptors1, descriptors2)  # Draw the top N matches

    # Sort the matches by distance (lower is better)

    matches_bf = sorted(matches_bf, key=lambda x: x.distance)

    # Draw the top N matches
    image_matches_bf = cv2.drawMatches(
        l_image, l_keypoints, r_image, r_keypoints, matches_bf, None
    )

    # show_image(image_matches_bf, "Matches", wait=True)

    tp = []  # target points

    qp = []  # query points

    for m in matches_bf:
        tp.append(r_keypoints[m.trainIdx].pt)
        qp.append(l_keypoints[m.queryIdx].pt)

    tp, qp = np.float32((tp, qp))

    # src_points = np.float32(
    #     [keypoints1[m.queryIdx].pt for m in matches_bf]
    # ).reshape(-1, 1, 2)
    # print("finding homography")
    homography, _ = cv2.findHomography(tp, qp, cv2.RANSAC, 5.0)

    # Print the estimated homography matrix

    # print("Estimated Homography Matrix:")

    # print(homography)

    # Warp the second image using the homography

    # We make the resulting image slightly wider to accommodate the new rotated image

    # print("warping perspective")
    result = cv2.warpPerspective(
        r_image, homography, (l_image.shape[1] + 800, l_image.shape[0])
    )

    # I bumped the final image size so I can see the whole paraonoma

    # Blending the warped image with the first image using alpha blending

    # First create a new image large enough to accommodate the stitched images

    # print("generating image")
    padded_left_img = cv2.copyMakeBorder(
        l_image,
        0,
        0,
        0,
        result.shape[1] - l_image.shape[1],
        cv2.BORDER_CONSTANT,
    )

    alpha = 0.5  # blending factor

    blended_image = cv2.addWeighted(padded_left_img, alpha, result, 1 - alpha, 0)

    # show_image(blended_image, "Blended Image", wait=True)

    return blended_image

    #
    #
    #
    #
    #
    #
    # AJBastroalign

    # convert keypoint to numpy array
    # In this case source is the right hand image
    # right_image is transformed to the target (left image)
    # source = np.asarray([[p.pt[0] + 10, p.pt[1] + 10] for p in r_keypoints])  # right
    # target = np.asarray([[p.pt[0] + 10, p.pt[1] + 10] for p in l_keypoints])  # left
    # # print(descriptors1)

    # transform, (src_pts, dst_pts) = aa.find_transform(
    #     source, target, max_control_points=250
    # )

    # plt.plot(src_pts[:, 0], 1240 - src_pts[:, 1], "xk", markersize=10)
    # plt.plot(dst_pts[:, 0], 1240 - dst_pts[:, 1], "og", markersize=10)

    # for i in range(len(src_pts)):
    #     plt.plot(
    #         [src_pts[i, 0], dst_pts[i, 0]],
    #         [1240 - src_pts[i, 1], 1240 - dst_pts[i, 1]],
    #         "-r",
    #     )

    # plt.show()

    # # To diplay the matrix you can use
    # # print("Transform", transform)

    # # and for its components
    # # print("Translation ", transform.translation)
    # # print("scale ", transform.scale)
    # # print("rotation angle ", transform.rotation)

    # # Convert transform to form OpenCV can use
    # homography = np.matrix(transform, np.float32)

    # # Apply transform to right image
    # result = cv2.warpPerspective(
    #     moon,
    #     homography,
    #     (earth.shape[1] + 400, earth.shape[0]),
    #     flags=cv2.INTER_LINEAR,
    # )

    # return result
