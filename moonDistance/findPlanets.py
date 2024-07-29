# Separate planets from background
# Save the image with the planets separated from the background
# into folder called separatedPlanets[time]

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from planets import Planet
import os
import shutil
    


def applyWhiteMask(image_path):
    # Load the image
    # Replace with the actual path to your image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

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




def separate_planet(image_path, save_path, show=1):
    # Load the image
    # Replace with the actual path to your image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)    
    whiteImg = applyWhiteMask(image_path)



    _, mask = cv2.threshold(whiteImg, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    # copy where we'll assign the new values
    green_hair = np.copy(img)
    # boolean indexing and assignment based on mask
    green_hair[(mask==255)] = [0,0,0]

    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(green_hair, cv2.COLOR_BGR2RGB))

    plt.show()

    # should be workin giwth green_hair instead



    # idea: first try to detect the starts by looking into white mask on the image


    # Convert the image to grayscale
    gray = cv2.cvtColor(green_hair, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help circle detection
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,  # inverse ratio of the accumulator resolution
        minDist=250,  # minimum distance between the centers of detected circles
        param1=10,  # gradient value for edge detection
        param2=20,  # accumulator threshold for circle detection
        minRadius=75,  # minimum radius of the detected circle
        maxRadius=120,  # maximum radius of the detected circle
    )

    listOfPlanetObjects = []

    # If circles are found, draw them on the image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            # cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            # cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
            # Need to create a new Planet object and save it to the list
            planet = Planet(i[0], i[1], i[2])
            planet.setFileName(fromPathGetFileName(image_path))
            listOfPlanetObjects.append(planet)

        # Display the result
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Detected Circles')
        # if show==1:
        #     plt.show()
        # Save the image using plt
        # plt.imsave(save_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # when key pressed, close the window
        plt.close()
        
    else:
        print("No circles detected.")

    print(f"Detected {len(listOfPlanetObjects)} planet[s].")
    return listOfPlanetObjects

def fromPathGetFileName(path):
    return os.path.basename(path)


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
    cropped_image = result[y:y+h, x:x+w]

    # Resize the cropped image to your desired dimensions
    desired_size = (300, 300)  # specify the dimensions you want
    resized_image = cv2.resize(cropped_image, desired_size)

    # Display or save the final result as needed
    cv2.imshow("Final Result", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return resized_image

# now a generic function
# it will, take the image path, the save path, and the planet object
# it will try to detect the planet in the image
# then it will remove the background and save the image to the save path
def planetDetection():


    input_path = sys.argv[1]
    output_path = sys.argv[1]

    listOfPlanetObjects = separate_planet(input_path, output_path, 1)

    # output path will be Planets/[name of the image].jpg


    # now I need to remove everything from the image except the planet
    # I will use the Planet object to do this
    # I will use the x, y, r to create a mask
    # then I will apply the mask to the image
    # then I will save the image to file
    counter = -1
    for planet in listOfPlanetObjects:
        counter += 1
        # read the image
        img = cv2.imread(output_path, cv2.IMREAD_COLOR)
        # create a mask
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(mask, (planet.x, planet.y), planet.r, (255, 255, 255), -1)
        # apply the mask
        result = cv2.bitwise_and(img, img, mask=mask)
        # crop it
        result = cropImage(result)
        # save the image
        # change the output path to include a counter so that the images are not overwritten
        # it should look like Planets/[name of the image]_[counter].jpg
        local_output_path = output_path.split(".")[0] + "_" + str(counter) + ".jpg"
        planet.setFileName(fromPathGetFileName(local_output_path))
        cv2.imwrite("circle_1.png", result)
    
    print("Planets separated and saved to Planets folder.")
    return listOfPlanetObjects

def earth_or_moon(circle, earth_template, moon_template, img):
    x, y, r = circle.x, circle.y, circle.r  # Assuming the class name for Planet is Planet.
    print("x:", x, "y:", y, "r:", r)
    # Define a confidence threshold for classification
    confidence_threshold = 0.95

    # Perform template matching with Earth template
    earth_result = cv2.matchTemplate(img, earth_template, cv2.TM_CCOEFF_NORMED)
    _, earth_confidence, _, _ = cv2.minMaxLoc(earth_result)

    # Perform template matching with Moon template
    moon_result = cv2.matchTemplate(img, moon_template, cv2.TM_CCOEFF_NORMED)
    _, moon_confidence, _, _ = cv2.minMaxLoc(moon_result)

    # print everything
    print(f"Earth confidence: {earth_confidence}")
    print(f"Moon confidence: {moon_confidence}")
    # print default confidence threshold
    print(f"Confidence threshold: {confidence_threshold}")

    # Find which confidence is higher
    # if both are below the threshold, return "Other"
    if earth_confidence < confidence_threshold and moon_confidence < confidence_threshold:
        return "Other"
    elif earth_confidence > moon_confidence:
        return "Earth"
    else:
        return "Moon"

def planetOfInterest():
    # Load the Earth and Moon templates
    earth_template = cv2.imread('Templates/earth.jpg', cv2.IMREAD_COLOR)
    moon_template = cv2.imread('Templates/moon.jpg', cv2.IMREAD_COLOR)



    # The list of planets are separated into the Planets folder so load them from there
    # clear the folder first
    detectIfFolderIsEmpty()
    # # load the planets from the image provided
    listOfPlanetObjects = planetDetection()
    # open the images in the Planets folder
    planets = []
    for filename in os.listdir('Planets'):
        # if the file does not contain a number, skip it
        if not any(char.isdigit() for char in filename):
            continue
        img = cv2.imread(f'Planets/{filename}', cv2.IMREAD_COLOR)
        print(f"Processing {filename}...")
        planets.append(img)
        # add the filename to the object
        


    # for each planet, determine if it is earth or moon
    counter = 0
    for planet in planets:
        output = earth_or_moon(listOfPlanetObjects[counter], earth_template, moon_template, planet)
        # print the result
        print(f"{listOfPlanetObjects[counter].fileName} is {output}.")
        
        listOfPlanetObjects[counter].setType(output)
        
        counter += 1

    # now i want to print a nice list of the planet's names and types
    print("Planet Name\tType")
    for planet in listOfPlanetObjects:
        print(f"{planet.fileName}\t{planet.type}")
    


def detectIfFolderIsEmpty():
    # If folder is not empty, remove all files from the folder
    folder = 'Planets'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
        
def main():
    planetDetection()
    # if we don't have earth/moon templates yet, create them
    # createTemplate()

    # planetOfInterest()



if __name__ == "__main__":
    main()
