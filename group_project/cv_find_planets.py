# Separate planets from background
# Save the image with the planets separated from the background
# into folder called separatedPlanets[time]

import cv2
import numpy as np
from .cv_planets import Planet
from .cv_detections import PlanetEnum
import os
from .cv_planet_model import predict_planet

# Load the Earth and Moon templates
earth_template = None
moon_template = None


def applyWhiteMask(img):
    # Load the image
    # Replace with the actual path to your image
    # # before detecting white, try to increase the brightness of the image
    # # convert the image to hsv
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # # split the channels
    # h, s, v = cv2.split(hsv)
    # # increase the brightness
    # v += 100
    # # merge the channels
    # final_hsv = cv2.merge((h, s, v))
    # # convert back to bgr
    # img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # create an image with only white pixels from the image
    # create a mask
    lower_white = np.array([200, 200, 200], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(img, lower_white, upper_white)

    # Apply the mask to the image
    result = cv2.bitwise_and(img, img, mask=white_mask)

    # i want to make every white in the image to be 255
    # and everything else to be 0
    # this will make it easier to detect the planets

    # Convert the image to grayscale
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to convert the image to binary
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # i want all the white pixels in the image to increase in size
    # so basically make a circle around each white pixel so it becomes bigger

    # Create a structuring element for the dilation
    kernel = np.ones((40, 40), np.uint8)

    # Apply dilation to increase the size of the white regions
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Display the result
    # plt.imshow(dilated, cmap='gray')
    # plt.title('Dilated Image')
    # save it
    # whiteMaskPath = "whiteMask.jpg"
    # plt.imsave(whiteMaskPath, dilated, cmap='gray')
    # plt.show()

    return dilated


def load_ros_templates():
    global earth_template, moon_template, low_q_earth_template, low_q_moon_template, very_low_q_earth_template, very_low_q_moon_template

    from ament_index_python.packages import get_package_share_directory

    package_path = get_package_share_directory("group_project")
    earth_path = os.path.join(package_path, "supporting_files", "earth.jpg")
    moon_path = os.path.join(package_path, "supporting_files", "earth.jpg")

    low_q_earth_path = os.path.join(package_path, "supporting_files", "low_q_earth.jpg")
    low_q_moon_path = os.path.join(package_path, "supporting_files", "low_q_moon.jpg")

    very_low_q_earth_path = os.path.join(
        package_path, "supporting_files", "very_low_q_earth.jpg"
    )
    very_low_q_moon_path = os.path.join(
        package_path, "supporting_files", "very_low_q_moon.jpg"
    )

    earth_template = cv2.imread(earth_path, cv2.IMREAD_COLOR)
    moon_template = cv2.imread(moon_path, cv2.IMREAD_COLOR)
    low_q_earth_template = cv2.imread(low_q_earth_path, cv2.IMREAD_COLOR)
    low_q_moon_template = cv2.imread(low_q_moon_path, cv2.IMREAD_COLOR)
    very_low_q_earth_template = cv2.imread(very_low_q_earth_path, cv2.IMREAD_COLOR)
    very_low_q_moon_template = cv2.imread(very_low_q_moon_path, cv2.IMREAD_COLOR)


def load_test_templates():
    global earth_template, moon_template, low_q_earth_template, low_q_moon_template, very_low_q_earth_template, very_low_q_moon_template

    earth_template = cv2.imread("./supporting_files/earth.jpg", cv2.IMREAD_COLOR)
    moon_template = cv2.imread("./supporting_files/moon.jpg", cv2.IMREAD_COLOR)

    # low quality earth moon
    low_q_earth_template = cv2.imread(
        "./supporting_files/low_q_earth.jpg", cv2.IMREAD_COLOR
    )
    low_q_moon_template = cv2.imread(
        "./supporting_files/low_q_moon.jpg", cv2.IMREAD_COLOR
    )

    # very low quality earth moon
    very_low_q_earth_template = cv2.imread(
        "./supporting_files/very_low_q_earth.jpg", cv2.IMREAD_COLOR
    )
    very_low_q_moon_template = cv2.imread(
        "./supporting_files/very_low_q_moon.jpg", cv2.IMREAD_COLOR
    )


def separate_planet(drawing, image):
    copy = image.copy()

    # Load the image
    # Replace with the actual path to your image
    whiteImg = applyWhiteMask(image)

    _, mask = cv2.threshold(whiteImg, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    # copy where we'll assign the new values
    green_hair = np.copy(image)
    # boolean indexing and assignment based on mask
    green_hair[(mask == 255)] = [0, 0, 0]

    # Convert the image to grayscale
    gray = cv2.cvtColor(green_hair, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help circle detection
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,  # inverse ratio of the accumulator resolution
        minDist=100,  # minimum distance between the centers of detected circles
        param1=40,  # gradient value for edge detection
        param2=20,  # accumulator threshold for circle detection
        minRadius=40,  # minimum radius of the detected circle
        maxRadius=100,  # maximum radius of the detected circle
    )

    listOfPlanetObjects = []

    # If circles are found, draw them on the image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(drawing, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # cv2.circle(copy, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(drawing, (i[0], i[1]), 2, (0, 0, 255), 3)
            # cv2.circle(copy, (i[0], i[1]), 2, (0, 0, 255), 3)
            # Need to create a new Planet object and save it to the list
            planet = Planet(i[0], i[1], i[2])
            listOfPlanetObjects.append(planet)

    # # show the image
    from .cv_detect import show

    # show(drawing, "detected circles", False, False)

    return listOfPlanetObjects, copy


def cropImage(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find contours in the image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the maximum area (assuming it represents the planet)
    max_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the planet using the maximum contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max_contour], 0, 255, thickness=cv2.FILLED)

    # Bitwise AND operation to apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)

    # Find the bounding box of the planet
    x, y, w, h = cv2.boundingRect(max_contour)

    # Crop the image to the bounding box
    cropped_image = result[y : y + h, x : x + w]

    # Resize the cropped image to your desired dimensions
    desired_size = (300, 300)  # specify the dimensions you want
    resized_image = cv2.resize(cropped_image, desired_size)

    return resized_image


# now a generic function
# it will, take the image path, the save path, and the planet object
# it will try to detect the planet in the image
# then it will remove the background and save the image to the save path
def planetDetection(drawing, input):
    listOfPlanetObjects, image = separate_planet(drawing, input)

    # output path will be Planets/[name of the image].jpg

    # now I need to remove everything from the image except the planet
    # I will use the Planet object to do this
    # I will use the x, y, r to create a mask
    # then I will apply the mask to the image
    # then I will save the image to file
    results = []
    counter = -1
    for planet in listOfPlanetObjects:
        counter += 1
        # read the image
        img = image.copy()
        # create a mask
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(mask, (planet.x, planet.y), planet.r, (255, 255, 255), -1)
        # apply the mask
        result = cv2.bitwise_and(img, img, mask=mask)
        # crop it
        result = cropImage(result)
        results.append(result)
        # save the image
        # change the output path to include a counter so that the images are not overwritten
        # it should look like Planets/[name of the image]_[counter].jpg

    return listOfPlanetObjects, results


def earth_or_moon(circle, img):

    planet, confidence = predict_planet(img)

    return planet, confidence

    x, y, r = (
        circle.x,
        circle.y,
        circle.r,
    )  # Assuming the class name for Planet is Planet.
    # Define a confidence threshold for classification
    confidence_threshold = 0.89

    # Perform template matching with Earth template
    earth_result = cv2.matchTemplate(img, earth_template, cv2.TM_CCOEFF_NORMED)
    _, earth_confidence, _, _ = cv2.minMaxLoc(earth_result)

    # Perform template matching with Moon template
    moon_result = cv2.matchTemplate(img, moon_template, cv2.TM_CCOEFF_NORMED)
    _, moon_confidence, _, _ = cv2.minMaxLoc(moon_result)

    # perform template matching with low quality earth and moon templates
    low_q_earth_result = cv2.matchTemplate(
        img, low_q_earth_template, cv2.TM_CCOEFF_NORMED
    )
    _, low_q_earth_confidence, _, _ = cv2.minMaxLoc(low_q_earth_result)

    low_q_moon_result = cv2.matchTemplate(
        img, low_q_moon_template, cv2.TM_CCOEFF_NORMED
    )
    _, low_q_moon_confidence, _, _ = cv2.minMaxLoc(low_q_moon_result)

    # perform template matching with very low quality earth and moon templates
    very_low_q_earth_result = cv2.matchTemplate(
        img, very_low_q_earth_template, cv2.TM_CCOEFF_NORMED
    )
    _, very_low_q_earth_confidence, _, _ = cv2.minMaxLoc(very_low_q_earth_result)

    very_low_q_moon_result = cv2.matchTemplate(
        img, very_low_q_moon_template, cv2.TM_CCOEFF_NORMED
    )
    _, very_low_q_moon_confidence, _, _ = cv2.minMaxLoc(very_low_q_moon_result)

    # print all the confidences
    # print("Earth confidence: ", earth_confidence)
    # print("Moon confidence: ", moon_confidence)
    # print("Low quality Earth confidence: ", low_q_earth_confidence)
    # print("Low quality Moon confidence: ", low_q_moon_confidence)
    # print("Very low quality Earth confidence: ", very_low_q_earth_confidence)
    # print("Very low quality Moon confidence: ", very_low_q_moon_confidence)
    # # print separator
    # print("=====================================")

    # 0.93 seemsto be the magic number

    # if any of the confidences are greater than 0.93 for all qualities, then we can easily
    # classify the planet

    if (
        earth_confidence > 0.93
        or low_q_earth_confidence > 0.93
        or very_low_q_earth_confidence > 0.93
    ):
        return PlanetEnum.EARTH, max(
            earth_confidence, low_q_earth_confidence, very_low_q_earth_confidence
        )

    if (
        moon_confidence > 0.93
        or low_q_moon_confidence > 0.93
        or very_low_q_moon_confidence > 0.93
    ):
        return PlanetEnum.MOON, max(
            moon_confidence, low_q_moon_confidence, very_low_q_moon_confidence
        )

    # else we will use the confidence threshold to classify the planet

    max_confidence = max(
        earth_confidence,
        moon_confidence,
        low_q_earth_confidence,
        low_q_moon_confidence,
        very_low_q_earth_confidence,
        very_low_q_moon_confidence,
    )

    # this code is VERY WEIRD @FELIX

    if max_confidence < confidence_threshold:
        return PlanetEnum.OTHER, max_confidence

    if earth_confidence == max_confidence:
        return PlanetEnum.EARTH, max_confidence

    return PlanetEnum.MOON, max_confidence


def planetOfInterest(drawing, image):

    # The list of planets are separated into the Planets folder so load them from there
    # clear the folder first
    # # load the planets from the image provided
    listOfPlanetObjects, planets = planetDetection(drawing, image)
    # open the images in the Planets folder

    # for each planet, determine if it is earth or moon
    counter = 0
    for planet in planets:
        output, confidence = earth_or_moon(listOfPlanetObjects[counter], planet)
        listOfPlanetObjects[counter].setType(output)
        listOfPlanetObjects[counter].confidence = confidence
        counter += 1

    # filtered = filter(lambda p: p.confidence > 0.25, listOfPlanetObjects)

    return listOfPlanetObjects


def main():
    # planetDetection()
    # if we don't have earth/moon templates yet, create them
    planetOfInterest()


if __name__ == "__main__":
    main()
