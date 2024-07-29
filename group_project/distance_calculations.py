import math
from .cv_detections import WindowDetection


def earth_distance(earth_detection: WindowDetection, logger):
    earth_radius = 12742
    distance = (3 * 2 * earth_radius * 720) / (2 * earth_detection.planet_data.r * 3)
    # logger.info(f'Earth r is {earth_detection.planet_data.r}')
    return distance


def moon_distance(moon_detection: WindowDetection):
    moon_radius = 3475
    distance = (3 * 2 * moon_radius * moon_detection.height) / (
        2 * moon_detection.planet_data.r * 3
    )
    return distance


def planets_distance(earth_distance, moon_distance):
    angle = math.acos(moon_distance / earth_distance)
    return moon_distance * math.tan(angle)
