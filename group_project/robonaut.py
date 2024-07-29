import threading
from group_project import coordinates
import rclpy
from rclpy.node import Node
import signal
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage
from math import sin, cos
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from .cv_compare import compare_images
from .cv_detections import (
    StatusDetection,
    PosterDetection,
    WindowDetection,
    StatusEnum,
    PlanetEnum,
    DetectionSwitch,
)
from .distance_calculations import earth_distance, moon_distance, planets_distance
import cv2
from geometry_msgs.msg import Twist  # Import Twist message type
from group_project.cv_detect import detect
from group_project.astro_stitch import astro_stitch
from group_project.hud import draw_hud, HudData, load_map
from datetime import datetime
import random
from .goals_and_actions import RoboGoals, RoboActions
from pathlib import Path

import time

CAMERA_WIDTH = 960


group_project_directory_path = Path.home() / "group30"
# measurements_file_path = "group30/measurements.txt"
measurements_file_path = str(group_project_directory_path / "measurements.txt")
view_earth_file_path = str(group_project_directory_path / "viewEarth.png")
view_moon_file_path = str(group_project_directory_path / "viewMoon.png")
panorama_file_path = str(group_project_directory_path / "panorama.png")
window_file_path = lambda num: str(group_project_directory_path / f"window{num}.png")


def create_group_project_directory():
    group_project_directory_path.mkdir(parents=True, exist_ok=True)


class RoboNaut(Node):
    def __init__(self):
        super().__init__("robotnaut")
        # Camera feed subscription
        self.camera_subscription = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10
        )

        # self.real_camera_subscription = self.create_subscription(
        #     CompressedImage, "/camera/image/compressed", self.image_callback, 10
        # )

        self.detection_switch = (
            DetectionSwitch()
            .set_detect_status_light(True)
            .set_detect_potential_windows(False)
        )

        self.declare_parameter("coordinates_file_path", "")
        coordinates_file_path = (
            self.get_parameter("coordinates_file_path")
            .get_parameter_value()
            .string_value
        )
        self.coordinates = coordinates.get_module_coordinates(coordinates_file_path)
        self.declare_parameter("module", "")
        self.module = self.get_parameter("module").get_parameter_value().string_value
        self.declare_parameter("place", "")
        self.place = self.get_parameter("place").get_parameter_value().string_value
        self.action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.cmd_vel_publisher = self.create_publisher(
            Twist, "/cmd_vel", 10
        )  # Assuming Twist message for velocity control
        self.current_goal_handle = None

        self.current_x = 0
        self.current_y = 0

        # Image detection attributes
        self.detections = []
        self.green_found = False
        self.red_found = False
        self.processing = False
        self.bridge = CvBridge()
        self.distance_from_green = 0
        self.distance_from_red = 0
        self.image_width = 0

        # Motion planning attributes
        self.current_action = None
        self.current_goal = None
        self.green_module = None
        self.current_position = None
        self.current_angle = None
        self.rate = self.create_rate(10)
        self.current_window = None
        self.module1_status = None
        self.module2_status = None
        self.start_time = datetime.now()
        self.earth_found = False
        self.moon_found = False
        self.earth_detection = None
        self.moon_detection = None
        self.current_action_result = None

        self.window_number = 1
        self.window_detections = []

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def get_coords(self):
        while True:
            try:
                # self.get_logger().info('Attempting to get position')
                pose = self.tf_buffer.lookup_transform(
                    "map",
                    "base_footprint",
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.1),
                )
                # self.get_logger().info(pose.transform.translation)
                break
            except Exception as e:
                pass

        self.current_position = pose.transform.translation
        self.current_angle = pose.transform.rotation

    def image_callback(self, data):
        try:
            # self.get_logger().info("Processing image...")
            # image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            drawing = image.copy()

            _, self.image_width, _ = image.shape

            detections, metadata = detect(drawing, image, self.detection_switch)

            self.detections = detections

            draw_hud(
                drawing,
                HudData(
                    detections=detections,
                    detect_metadata=metadata,
                    x=0,
                    y=0,
                    current_goal=self.current_goal,
                    module1_status=self.module1_status,
                    module2_status=self.module2_status,
                    start_time=self.start_time,
                ),
            )

            window_name = "camera_Feed"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, drawing)
            cv2.resizeWindow(window_name, 320, 240)
            key = cv2.waitKey(3)

            return
        except Exception as e:
            # print(e)
            pass
        finally:
            return

    def send_goal(self, x, y, yaw):

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # Orientation
        goal_msg.pose.pose.orientation.z = sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2)

        self.action_client.wait_for_server()
        result = self.action_client.send_goal(goal_msg)
        return result

    def stop_motion(self):
        desired_velocity = Twist()
        self.cmd_vel_publisher.publish(desired_velocity)

    # Method which checks for the precense of windows
    def check_windows(self):
        # Check for windows during motion
        for detection in self.detections:
            if isinstance(detection, WindowDetection):
                if detection.planet == PlanetEnum.EARTH:
                    if self.earth_found == False:
                        self.current_goal = RoboGoals.EARTH_FOUND_MOVE_TO_WINDOW
                        self.current_window = detection
                        self.current_action = RoboActions.MOVE_TO_WINDOW
                        return True
                    else:
                        self.current_goal = RoboGoals.EARTH_FOUND_ALREADY_ROTATING
                elif detection.planet == PlanetEnum.MOON:
                    if self.moon_found == False:
                        self.current_goal = RoboGoals.MOON_FOUND_MOVE_TO_WINDOW
                        self.current_window = detection
                        self.current_action = RoboActions.MOVE_TO_WINDOW
                        return True
                    else:
                        self.current_goal = RoboGoals.MOON_FOUND_ALREADY_ROTATING
                else:
                    self.current_window = detection
                    self.current_action = RoboActions.MOVE_TO_WINDOW
                    self.current_goal = RoboGoals.WINDOW_FOUND
                    self.get_logger().info(f"Found a window with other planets.")
                    return True
            elif isinstance(detection, PosterDetection):
                self.current_window = detection
                self.current_action = RoboActions.MOVE_TO_WINDOW
                self.current_goal = RoboGoals.WINDOW_FOUND
                self.get_logger().info(f"Found a poster.")
                return False

    # Method which checks for the green circle and its size
    def check_green_circle(self):
        for detection in self.detections:
            # switch case of instance of detection
            detection.x
            detection.y
            detection.size  # determine how far away?
            if isinstance(detection, StatusDetection):
                if detection.status == StatusEnum.GREEN:
                    self.green_found = True
                    self.distance_from_green = detection.size
                    self.get_logger().info("Green circle found.")
                elif detection.status == StatusEnum.RED:
                    self.red_found = True
                    self.distance_from_red = detection.size
                    self.get_logger().info("Red circle found.")

    # Method which rotates the robot in place. Further calls check_green_circle
    def rotate(self, time):
        desired_velocity = Twist()
        desired_velocity.angular.z = math.pi / 6

        self.green_found = False
        self.red_found = False

        for i in range(time):
            self.cmd_vel_publisher.publish(desired_velocity)
            self.rate.sleep()
            if self.green_module == None:
                self.check_green_circle()
            elif self.check_windows():
                desired_velocity = Twist()
                self.cmd_vel_publisher.publish(desired_velocity)
                break

    def rotate_inverse(self, time):
        desired_velocity = Twist()
        desired_velocity.angular.z = -(math.pi / 6)

        self.green_found = False
        self.red_found = False

        for i in range(time):
            self.cmd_vel_publisher.publish(desired_velocity)
            self.rate.sleep()
            if self.green_module == None:
                self.check_green_circle()
            elif self.check_windows():
                desired_velocity = Twist()
                self.cmd_vel_publisher.publish(desired_velocity)
                break

    # Method for motion planning in order to find the correct windows
    def find_windows(self):
        # Some parameters set
        max_radius = 2.5
        self.current_action = RoboActions.FIND_WINDOWS

        # The starting point of random sampling is the module's centre
        start_x = self.green_module.center.x
        start_y = self.green_module.center.y

        normal = True

        # Search the room (starting from the center) in a spiral form
        while True:
            # Counter set for 10 random samples
            counter = 0

            # Random distance from center
            radius_x = random.uniform(-max_radius, max_radius)
            radius_y = random.uniform(-max_radius, max_radius)
            orientation = random.uniform(0, 2 * math.pi)

            # Calculdesired_velocity = Twist()ate the next position to set as goal in the spiral
            next_x = start_x + radius_x
            next_y = start_y + radius_y
            next_orientation = orientation

            # Set a goal to the action client so it starts moving to it
            self.send_goal(next_x, next_y, next_orientation)
            self.current_goal = RoboGoals.ROTATE_FIND_WINDOWS

            if normal:
                self.rotate(140)
                normal = False
            else:
                self.rotate_inverse(140)
                normal = True

            if self.current_action == RoboActions.MOVE_TO_WINDOW:
                self.move_towards_window()
                self.current_action = RoboActions.FIND_WINDOWS

            self.current_goal = RoboGoals.NAVIGATE_FIND_WINDOWS
            counter += 1

            # If a certain amount was searched
            if counter == 5:
                self.get_logger().info("Increasing the search space...")
                max_radius = 3
            elif self.earth_found and self.moon_found:
                self.current_goal = RoboGoals.COMPUTING_MEASUREMENTS
                return

    def rotate_to_angle(self, angular_velocity):
        try:
            # Set the direction of rotation based on the sign of the angular velocity
            desired_velocity = Twist()
            if angular_velocity < 0:
                # Rotate left (counterclockwise)
                desired_velocity.angular.z = 0.1  # Set left rotation velocity
            else:
                # Rotate right (clockwise)
                desired_velocity.angular.z = -0.1  # Set right rotation velocity

            # Publish the desired velocity command
            self.cmd_vel_publisher.publish(desired_velocity)
            return True
        except:
            self.get_logger().info(
                "Something wrong with getting the angular velocity..."
            )
            return False

    def get_most_desired_window(self):
        """
        gets the window detection thats most centred within the camera feed
        """

        window_detections = filter(
            lambda detection: isinstance(detection, WindowDetection), self.detections
        )
        detections_with_diff = [
            (detection, abs(self.image_width - detection.x))
            for detection in window_detections
        ]
        detections_with_diff_sorted = list(
            sorted(detections_with_diff, key=lambda data: data[1])
        )

        if len(detections_with_diff_sorted) > 0:
            return detections_with_diff_sorted[0][0]

        return None

    # Method which gets the current angle of rotatino between the robot and the window
    def get_current_angle(self):

        most_desired_detection = self.get_most_desired_window()

        if most_desired_detection is None:
            return None

        target_x = most_desired_detection.x
        camera_x = self.image_width

        fov_degree = 62.200011761
        pixels_per_degree = camera_x / fov_degree

        camera_center_x = camera_x / 2

        target_offset_x = target_x - camera_center_x
        target_offset_deg = target_offset_x / pixels_per_degree

        return target_offset_deg

    # moving towards window
    def move_towards_window(self):

        # start the timer for the function
        # the function will currently timeout after 30 seconds
        start_time = time.time()

        target_offset_deg = self.get_current_angle()

        # self.get_logger().info("Rotating towards window...")
        if self.rotate_to_angle(target_offset_deg) == False:
            return

        # self.get_logger().info(f"checking TOD : {target_offset_deg}")

        # # Wait for the rotation to complete before moving forward
        # self.get_logger().info("Waiting for rotation to complete...")

        # Add debugging logs
        while True:
            current_angle = self.get_current_angle()
            if current_angle == None:
                time.sleep(0.1)
                continue

            if self.rotate_to_angle(target_offset_deg) == False:
                return
            # self.get_logger().info(f"checking angle : {current_angle}")

            if math.isclose(current_angle, 0, abs_tol=2.00):
                # self.get_logger().info("Angle alignment achieved.")
                break

            if time.time() - start_time > 10:  # Check if 10 seconds have passed
                self.get_logger().warn("Rotation towards window timed out.")
                return

            # self.get_logger().info("Waiting for angle alignment...")
            # self.get_logger().info(
            #     f"Current angle: {current_angle}, Target angle: {target_offset_deg}"
            # )
            time.sleep(0.1)

        # Move forward towards the window with initial velocity
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.2  # Initial forward velocity
        self.cmd_vel_publisher.publish(desired_velocity)

        robot_stopped = False

        # Monitor distance to the window until stopping distance is reached
        while robot_stopped == False:

            for detection in self.detections:
                if isinstance(detection, WindowDetection):

                    for window in self.window_detections:
                        if compare_images(window, detection.capture) > 0.45:
                            self.get_logger().info(
                                "Window already captured. Ignoring..."
                            )
                            return

                    if time.time() - start_time > 40:  # Check if 30 seconds have passed
                        self.get_logger().warn("Capturing window timed out.")
                        desired_velocity = Twist()
                        self.cmd_vel_publisher.publish(desired_velocity)
                        time.sleep(0.1)
                        return

                    # Adjust stopping threshold based on distance to the window
                    if detection.size > 75000 and (
                        True if detection.planet_data else True
                    ):
                        self.get_logger().info("Stopping near window...")
                        if (
                            detection.planet == PlanetEnum.EARTH
                            and self.earth_detection == None
                        ):
                            self.earth_detection = detection
                            self.get_logger().info("Capturing Earth.")
                            cv2.imwrite(view_earth_file_path, detection.capture)
                            self.earth_found = True
                        elif (
                            detection.planet == PlanetEnum.MOON
                            and self.moon_detection == None
                        ):
                            self.moon_detection = detection
                            self.get_logger().info("Capturing Moon.")
                            cv2.imwrite(view_moon_file_path, detection.capture)
                            self.moon_found = True

                        self.get_logger().info("Capturing the window.")
                        cv2.imwrite(
                            window_file_path(self.window_number),
                            detection.capture,
                        )
                        self.window_number += 1
                        self.window_detections.append(detection.capture)
                        desired_velocity = Twist()
                        self.cmd_vel_publisher.publish(desired_velocity)
                        time.sleep(2.0)
                        return
                    else:
                        desired_velocity = Twist()
                        desired_velocity.linear.x = 0.2
                        self.cmd_vel_publisher.publish(desired_velocity)
                        time.sleep(0.1)

            no_detection_time = time.time()

            while len(self.detections) == 0:
                time.sleep(0.1)
                if (time.time() - no_detection_time) > 2:
                    self.get_logger().info(
                        "Cannot find the window anymore. Aborting..."
                    )
                    desired_velocity = Twist()
                    self.cmd_vel_publisher.publish(desired_velocity)
                    return
                if detection.size > 70000:
                    desired_velocity = Twist()
                    self.cmd_vel_publisher.publish(desired_velocity)
                    self.get_logger().info("Ignoring this poster...")
                    return
            time.sleep(0.1)

        desired_velocity = Twist()
        self.cmd_vel_publisher.publish(desired_velocity)

    def compute_distances(self):
        self.get_logger().info(
            f"Calculating the distances between the planets and the spaceship"
        )

        with open(measurements_file_path, "w") as file:

            # Earth computation:
            edistance = earth_distance(self.earth_detection, self.get_logger())
            file.writelines(f"Earth: {edistance} km\n")

            self.get_logger().info(f"earth distance: {edistance}")

            # Moon computation:
            mdistance = moon_distance(self.moon_detection)
            file.writelines(f"Moon: {mdistance} km\n")

            self.get_logger().info(f"moon distance: {mdistance}")

            # Distance between the two:
            bdistance = planets_distance(edistance, mdistance)

            file.writelines(f"Distance: {bdistance} km\n")

            self.get_logger().info(f"Distance between planets: {bdistance}")

            panorama = astro_stitch(
                self.earth_detection.capture, self.moon_detection.capture
            )

            self.get_logger().info("Stitching the images...")
            cv2.imwrite(panorama_file_path, panorama)

    # Method which would be called in order to find the green module/room at the beginning
    def find_green_room(self):
        # Set the current action as finding the green room
        self.current_goal = RoboGoals.POSE_WAITING
        self.get_coords()

        # Wait for the process to return the robot's current position
        while self.current_position == None:
            continue

        # Check which module's entrance is closer to the current robot's position:
        if math.dist(
            [self.current_position.x, self.current_position.y],
            [
                self.coordinates.module_1.entrance.x,
                self.coordinates.module_1.entrance.y,
            ],
        ) < math.dist(
            [self.current_position.x, self.current_position.y],
            [
                self.coordinates.module_2.entrance.x,
                self.coordinates.module_2.entrance.y,
            ],
        ):
            self.current_action = RoboActions.MODULE_1_ENTRANCE
            # Set the current action as finding the green room
            self.current_goal = RoboGoals.MODULE_1_ENTRANCE_NAVIGATING
            self.send_goal(
                self.coordinates.module_1.entrance.x,
                self.coordinates.module_1.entrance.y,
                0,
            )

        else:
            self.current_action = RoboActions.MODULE_2_ENTRANCE
            # Set the current action as finding the green room
            self.current_goal = RoboGoals.MODULE_2_ENTRANCE_NAVIGATING
            self.send_goal(
                self.coordinates.module_2.entrance.x,
                self.coordinates.module_2.entrance.y,
                0,
            )

        # Rotate in place to find the modules
        self.current_goal = (
            RoboGoals.MODULE_1_ENTRANCE_ROTATING
            if self.current_action == RoboActions.MODULE_1_ENTRANCE
            else RoboGoals.MODULE_2_ENTRANCE_ROTATING
        )
        self.rotate(140)

        # Check whether the green button was found
        if (
            self.green_found == True
            and self.distance_from_green > self.distance_from_red
        ):
            if self.current_action == RoboActions.MODULE_1_ENTRANCE:
                self.green_module = self.coordinates.module_1
                self.module1_status = StatusEnum.GREEN
                self.module2_status = StatusEnum.RED
            else:
                self.green_module = self.coordinates.module_2
                self.module1_status = StatusEnum.RED
                self.module2_status = StatusEnum.GREEN

        else:
            if self.current_action == RoboActions.MODULE_1_ENTRANCE:
                self.green_module = self.coordinates.module_2
                self.module1_status = StatusEnum.RED
                self.module2_status = StatusEnum.GREEN

            else:
                self.green_module = self.coordinates.module_1
                self.module1_status = StatusEnum.GREEN
                self.module2_status = StatusEnum.RED

        self.current_goal = RoboGoals.NAVIGATE_FIND_WINDOWS
        self.detection_switch.set_detect_status_light(
            False
        ).set_detect_potential_windows(True)


def main():
    from .cv_find_planets import load_ros_templates

    create_group_project_directory()
    load_ros_templates()

    # from ament_index_python.packages import get_package_share_directory
    # import os

    # package_path = get_package_share_directory("group_project")
    # map_path = os.path.join(package_path, "worlds", "spacecraft_easy", "map")

    # load_map(map_path)

    def signal_handler(sig, frame):
        robonaut.cancel_goal()
        rclpy.shutdown()

    rclpy.init(args=None)
    robonaut = RoboNaut()

    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(robonaut,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            # Initial motion, check whether the robot has an underlying action and goal
            if robonaut.current_action == None and robonaut.current_goal == None:
                robonaut.find_green_room()

            elif robonaut.current_goal == RoboGoals.NAVIGATE_FIND_WINDOWS:
                robonaut.find_windows()

            elif robonaut.current_goal == RoboGoals.COMPUTING_MEASUREMENTS:
                robonaut.compute_distances()
                robonaut.stop_motion()
                break
            else:
                pass

        robonaut.stop_motion()

    except Exception as e:
        robonaut.get_logger().info(e)
        robonaut.stop_motion()
        import traceback

        # print(traceback.format_exc())


if __name__ == "__main__":
    main()
