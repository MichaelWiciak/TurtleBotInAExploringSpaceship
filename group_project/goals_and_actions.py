from enum import Enum


class RoboGoals(Enum):
    """
    Goals is used for displaying in the HUD
    """
    WINDOW_FOUND = "Found a window. Moving towards it."
    EARTH_FOUND_MOVE_TO_WINDOW = "Found earth. Moving to Window."
    EARTH_FOUND_ALREADY_ROTATING = "Found earth but already captured. Rotating..."
    MOON_FOUND_MOVE_TO_WINDOW = "Found moon. Moving to Window."
    MOON_FOUND_ALREADY_ROTATING = "Found moon but already captured. Rotating..."
    POSTER_DETECTED_IGNORE = "Poster detected. Ignoring..."
    ROTATE_FIND_WINDOWS = "Rotating to find Windows"
    POSE_WAITING = "Waiting for the current pose of the robot..."
    MODULE_1_ENTRANCE_NAVIGATING = "Navigating to Module 1's entrance"
    MODULE_2_ENTRANCE_NAVIGATING = "Navigating to Module 2's entrance"
    MODULE_1_ENTRANCE_ROTATING = "Rotating at Module 1's entrance"
    MODULE_2_ENTRANCE_ROTATING = "Rotating at Module 2's entrance"
    COMPUTING_MEASUREMENTS = "Moon and Earth found. Computing measurements..."
    NAVIGATE_FIND_WINDOWS = "Navigating to find Windows"

    def __str__(self) -> str:
        return self.value


class RoboActions(Enum):
    """
    Actions are used internally for logic
    """

    MOVE_TO_WINDOW = "Move to Window"
    FIND_WINDOWS = "Find Windows"
    MODULE_1_ENTRANCE = "Module 1 Entrance"
    MODULE_1_CENTER = "Module 1 Center"
    MODULE_2_ENTRANCE = "Module 2 Entrance"
    MODULE_2_CENTER = "Module 2 Center"

    def __str__(self) -> str:
        return self.value
