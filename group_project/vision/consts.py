import numpy as np


sensitivity = 10

hsv_red_lower = np.array([0 - sensitivity, 100, 100])
hsv_red_upper = np.array([0 + sensitivity, 255, 255])
hsv_green_lower = np.array([60 - sensitivity, 100, 100])
hsv_green_upper = np.array([60 + sensitivity, 255, 255])
hsv_blue_lower = np.array([120 - sensitivity, 100, 100])
hsv_blue_upper = np.array([120 + sensitivity, 255, 255])
