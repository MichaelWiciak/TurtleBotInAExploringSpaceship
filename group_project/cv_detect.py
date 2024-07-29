import cv2
import os
import numpy as np
from time import time
from .cv_detections import (
    PosterDetection,
    StatusDetection,
    WindowDetection,
    StatusEnum,
    PlanetEnum,
    DetectMetadata,
    DetectionSwitch,
)
from .four_point_transform import four_point_transform
from .cv_find_planets import planetOfInterest
from .hud import draw_hud, HudData, load_map


debug_show = False

colour_map = {
    "green": (0, 255, 0),
    "red": (38, 38, 220),
    "outline": (238, 211, 34),
    "outline2": (235, 99, 37),
    "text": (0, 0, 0),
    "text_drop": (255, 255, 255),
}


outline_thickness = 2


def rect_in_rect(rect1, rect2) -> bool:
    # modified from https://stackoverflow.com/a/62722520/13121213
    r1_x1, r1_y1, r1_w, r1_h = rect1
    r2_x1, r2_y1, r2_w, r2_h = rect2

    r1_x2 = r1_x1 + r1_w
    r1_y2 = r1_y1 + r1_h

    r2_x2 = r2_x1 + r2_w
    r2_y2 = r2_y1 + r2_h

    if r1_x1 > r2_x1 and r1_x2 < r2_x2 and r1_y1 > r2_y1 and r1_y2 < r2_y2:
        return True

    return False


# modified from https://github.com/ashwin-pajankar/Python_Courses_and_Tutorials/blob/main/Course_004_Image_Processing_and_Computer_Vision/02_OpenCV_Computer_Vision/Section%2022/05_MaxRGB.ipynb
#
# modification: maximise the colours
def max_bgr_filter(image):
    (r, g, b) = cv2.split(image)
    maximum = cv2.max(cv2.max(r, g), b)
    minimum = cv2.min(cv2.min(r, g), b)
    grey = (maximum - minimum) < 32

    r[grey] = 0
    g[grey] = 0
    b[grey] = 0

    r[r < maximum] = 0
    g[g < maximum] = 0
    b[b < maximum] = 0

    r[r >= maximum] = 255
    g[g >= maximum] = 255
    b[b >= maximum] = 255

    return cv2.merge([r, g, b])


def find_image_paths():
    assets_dir = "./assets/images"
    image_paths = []

    for dirpath, _dirnames, filenames in os.walk(assets_dir):
        for filename in filenames:
            if filename.endswith(".png"):
                image_paths.append(os.path.join(dirpath, filename))

    return image_paths


def find_video_paths():
    assets_dir = "./assets/recordings"
    video_paths = []

    for dirpath, _dirnames, filenames in os.walk(assets_dir):
        for filename in filenames:
            if filename.endswith(".avi"):
                video_paths.append(os.path.join(dirpath, filename))

    return video_paths


def detect_windows_and_posters(
    drawing, image
) -> list[WindowDetection | PosterDetection]:
    # window frames always has white borders, and are rectangular
    detections: list[WindowDetection | PosterDetection] = []

    grey = cv2.cvtColor(
        cv2.convertScaleAbs(image, alpha=1.5, beta=1.5), cv2.COLOR_BGR2GRAY
    )

    ret, thresh = cv2.threshold(grey, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html

    rect_contours = []

    # rectangle detection inspired from
    # https://www.tutorialspoint.com/how-to-detect-a-rectangle-and-square-in-an-image-using-opencv-python
    for contour, c_hierarchy in zip(
        contours, hierarchy[0] if hierarchy is not None else []
    ):
        c_prev, c_next, c_child, c_parent = c_hierarchy
        # window always has a parent "frame"
        if c_parent == -1:
            continue

        window_area = cv2.contourArea(contour)
        if window_area < 50:
            continue
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

        _, _, w, h = cv2.boundingRect(contour)

        # window frames are always rectangular
        if w < 50 or h < 50:
            continue
        if w / h < 0.5:
            continue

        if len(approx) != 4:
            continue

        # area should be similar to the bounding rect
        bounding_area = w * h
        if bounding_area / window_area > 2:
            continue

        rect_contours.append(contour)
        cv2.drawContours(
            drawing, [approx], 0, colour_map["outline2"], outline_thickness
        )

        M = cv2.moments(contour)
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        capture = four_point_transform(image, approx.reshape(4, 2))
        capture_drawing = capture.copy()

        planets = planetOfInterest(capture_drawing, capture)

        label = ""

        if len(planets) == 0:
            label = "Window: No planets"
            detections.append(
                WindowDetection(
                    cx,
                    cy,
                    window_area,
                    h,
                    PlanetEnum.NOT_DETECTED,
                    None,
                    capture=capture,
                    capture_drawing=capture_drawing,
                )
            )

        elif len(planets) == 1:
            confidence = planets[0].confidence * 100
            label = f"Window: {planets[0].type} ({confidence:.2f}%)"
            detections.append(
                WindowDetection(
                    cx,
                    cy,
                    window_area,
                    h,
                    planets[0].type,
                    planets[0],
                    capture=capture,
                    capture_drawing=capture_drawing,
                )
            )
        else:
            label = "Poster"
            detections.append(
                PosterDetection(
                    cx,
                    cy,
                    window_area,
                    capture=capture,
                    capture_drawing=capture_drawing,
                )
            )

        cv2.putText(
            drawing,
            label,
            (cx, cy),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            colour_map["outline2"],
            outline_thickness,
        )

    cv2.drawContours(
        drawing, rect_contours, -1, colour_map["outline2"], outline_thickness
    )
    return detections


def detect_module_status_light(drawing, image) -> list[StatusDetection]:
    """
    Status lights are always circular, either in green or red. and are always contained
    within a white rectangle
    """

    detections: list[StatusDetection] = []

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    max_bgr_image = max_bgr_filter(image)

    # show(max_bgr_image, "max_bgr_image", False, wait=False)

    bgr_red_lower = np.array([0, 0, 250])
    bgr_red_upper = np.array([0, 0, 255])

    bgr_green_lower = np.array([0, 250, 0])
    bgr_green_upper = np.array([0, 255, 0])

    red_mask = cv2.inRange(max_bgr_image, bgr_red_lower, bgr_red_upper)
    green_mask = cv2.inRange(max_bgr_image, bgr_green_lower, bgr_green_upper)
    red_only = cv2.bitwise_and(max_bgr_image, max_bgr_image, mask=red_mask)
    green_only = cv2.bitwise_and(max_bgr_image, max_bgr_image, mask=green_mask)

    grey_red = cv2.cvtColor(red_only, cv2.COLOR_BGR2GRAY)
    ret, thresh_red = cv2.threshold(grey_red, 255, 255, 255)
    contours_red, h = cv2.findContours(
        thresh_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    grey_green = cv2.cvtColor(green_only, cv2.COLOR_BGR2GRAY)
    ret, thresh_green = cv2.threshold(grey_green, 255, 255, 255)
    contours_green, h = cv2.findContours(
        thresh_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    global debug_show
    debug_show = True

    rect_contours = []
    red_contours = []
    green_contours = []

    def filter_contours(contour):
        return cv2.contourArea(contour) > 50

    contours_green = [contour for contour in contours_green if filter_contours(contour)]
    contours_red = [contour for contour in contours_red if filter_contours(contour)]

    # rectangle detection inspired from
    # https://www.tutorialspoint.com/how-to-detect-a-rectangle-and-square-in-an-image-using-opencv-python

    for status, status_contours, circle_contours in [
        ("red", contours_red, red_contours),
        ("green", contours_green, green_contours),
    ]:
        for status_contour in status_contours:
            actual_area = cv2.contourArea(status_contour)

            M = cv2.moments(status_contour)
            status_x, status_y = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

            bounding_status = cv2.boundingRect(status_contour)

            # # if center not similar, then probably notstatus light
            # if abs(rect_x - status_x) > 20 or abs(rect_rect_y - status_y) > 20:
            #     continue

            # check if the area of contour matches the enclosing circle area
            # if not probably not a circle
            (x, y), radius = cv2.minEnclosingCircle(status_contour)
            expected_area = np.pi * radius**2
            if (abs(actual_area - expected_area) / expected_area) > 0.5:
                continue

            circle_contours.append(status_contour)

            detections.append(
                StatusDetection(
                    status_x,
                    status_y,
                    actual_area,
                    StatusEnum.RED if status == "red" else StatusEnum.GREEN,
                )
            )

            cv2.putText(
                drawing,
                f"Status {status}",
                (status_x, status_y),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                colour_map[status],
                outline_thickness,
            )

    cv2.drawContours(
        drawing, rect_contours, -1, colour_map["outline"], outline_thickness
    )
    cv2.drawContours(
        drawing, red_contours, -1, colour_map["outline"], outline_thickness
    )
    cv2.drawContours(
        drawing, green_contours, -1, colour_map["outline"], outline_thickness
    )

    # cv2.drawContours(
    #     drawing, contours_white, -1, colour_map["outline"], outline_thickness
    # )
    # cv2.drawContours(
    #     drawing, contours_green, -1, colour_map["outline"], outline_thickness
    # )
    # cv2.drawContours(
    #     drawing, contours_red, -1, colour_map["outline"], outline_thickness
    # )
    return detections


# detecting everything by default
default_detect = DetectionSwitch()


def detect(drawing, image, switch=default_detect):
    detections: list[StatusDetection | WindowDetection | PosterDetection] = []

    if switch.detect_potential_windows:
        window_start = time()
        window_detections = detect_windows_and_posters(drawing, image)
        window_end = time()
        window_took = window_end - window_start
        detections += window_detections
    else:
        window_took = 0

    if switch.detect_status_light:
        status_light_start = time()
        status_light_detections = detect_module_status_light(drawing, image)
        status_light_end = time()
        status_light_took = status_light_end - status_light_start
        detections += status_light_detections
    else:
        status_light_took = 0

    return detections, DetectMetadata(window_took, status_light_took)


def show(image, name="image", destroy=True, wait=True):
    if not debug_show:
        return
    cv2.imshow(name, image)
    cv2.resizeWindow(name, 960, 720)
    if wait:
        cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()


def main():
    from datetime import datetime, timedelta

    global debug_show
    debug_show = True

    import argparse
    from .cv_find_planets import load_test_templates

    load_test_templates()
    load_map("worlds/spacecraft_hard/map")

    parser = argparse.ArgumentParser(description="Detect windows and status lights")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-step", action="store_true")
    parser.add_argument("--video-save", action="store_true")
    parser.add_argument("--image", action="store_true")
    parser.add_argument("--local", action="store_true")

    args = parser.parse_args()

    print(args)

    if args.local:
        from .cv_planet_model import set_local_model

        set_local_model(True)

    if args.video:
        print("Starting video processing...")
        video_paths = find_video_paths()
        for video_path in video_paths:
            start_time = datetime.now()
            filename = os.path.basename(video_path)

            vid = cv2.VideoCapture(video_path)
            out: cv2.VideoWriter | None = None

            while vid.isOpened():
                success, frame = vid.read()

                # get dimensions of the frame
                if out is None and args.video_save:
                    height, width, _ = frame.shape
                    out = cv2.VideoWriter(
                        f"{filename}.processed.{time()}.avi",
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        10,
                        (960, 720),
                    )

                if success:  # frame read successfully

                    start_time = start_time - timedelta(milliseconds=50)

                    drawing = frame.copy()
                    cv2.putText(
                        drawing,
                        f"Path: {video_path}",
                        (10, 10),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 0),
                        2,
                    )
                    detections, metadata = detect(drawing, frame)

                    drawing = cv2.resize(drawing, (960, 720))
                    draw_hud(
                        drawing,
                        HudData(
                            detections=detections,
                            detect_metadata=metadata,
                            module1_status=StatusEnum.RED,
                            module2_status=StatusEnum.GREEN,
                            current_goal="Move to Green Room",
                            x=3.14159265359,
                            y=2.71828182846,
                            start_time=start_time,
                        ),
                    )

                    if out is not None:
                        out.write(drawing)

                    if args.video_step:
                        show(drawing, "drawing", False)

                    pass
                else:
                    break

            if out is not None:
                out.release()
            vid.release()

    if args.image:
        print("Starting image processing...")
        image_paths = find_image_paths()
        images = [(image_path, cv2.imread(image_path)) for image_path in image_paths]

        for image_path, image in images:
            drawing = image.copy()
            cv2.putText(
                drawing,
                f"Path: {image_path}",
                (10, 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 0),
                2,
            )

            detections, metadata = detect(drawing, image)
            drawing = cv2.resize(drawing, (960, 720))
            draw_hud(
                drawing,
                HudData(
                    detections=detections,
                    detect_metadata=metadata,
                    module1_status=StatusEnum.RED,
                    module2_status=StatusEnum.GREEN,
                    current_goal="Move to Green Room",
                    x=3.14159265359,
                    y=2.71828182846,
                    start_time=datetime.now(),
                ),
            )
            show(drawing, "drawing", False)

    print("Done.")
