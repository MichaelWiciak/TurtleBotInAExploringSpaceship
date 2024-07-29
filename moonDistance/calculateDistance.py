# Takes the 2d coordinates from detectMoon.py and calculates the distance between the two points
# then takes that distance and using the actual distance between the two planets, earth and moon stored in variable and 
# calculates the distance between the two planets.

import math
import numpy as np
import cv2
import time
import sys
import os

# Function to calculate the distance between two points
def calculateDistance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to calculate the distance between the two planets
def calculateMoonDistance(distance):
    # Actual distance between the two planets
    actualDistance = 384400 # in km
    # Distance between the two points
    distanceBetweenPoints = 0.0
    # Distance between the two planets
    distanceBetweenPlanets = 0.0
    # Distance between the two points
    distanceBetweenPoints = distance * 384400 / 1000
    # Distance between the two planets
    distanceBetweenPlanets = math.sqrt((distanceBetweenPoints**2) - (actualDistance**2))
    return distanceBetweenPlanets