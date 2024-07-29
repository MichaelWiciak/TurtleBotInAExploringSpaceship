# Modified from lab 5 Exercise 1

# from __future__ import division
import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.exceptions import ROSInterruptException
import signal
from cv_bridge import CvBridge, CvBridgeError
from time import time
from .cv_detect import detect
from .cv_detections import StatusDetection, PosterDetection, WindowDetection
from .hud import draw_hud, HudData, load_map
from geometry_msgs.msg import PoseWithCovarianceStamped
from datetime import datetime

start_time = datetime.now()

out = cv2.VideoWriter(
    f"recording.{time()}.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, (960, 720)
)


class CameraFeed(Node):
    def __init__(self):
        super().__init__("camera_feed_viewer")
        self.subscription = self.create_subscription(
            Image, "/camera/image_raw", self.callback, 10
        )
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self.pose_callback, 10
        )
        self.pose_data = None

        self.bridge = CvBridge()
        self.subscription  # prevent unused variable warning

        self.sensitivity = 10

        self.processing = False
        self.detection_on = False
        self.recording = False

    def get_coords(self):
        if self.pose_data is None:
            return 0, 0

        x = self.pose_data.pose.pose.position.x
        y = self.pose_data.pose.pose.position.y

        return x, y

    def pose_callback(self, data):
        self.pose_data = data

    def callback(self, data):
        if self.processing:
            self.get_logger().info("Already processing image, skipping...")
            return
        try:
            self.processing = True
            self.get_logger().info("Processing image...")
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            drawing = image.copy()

            if self.detection_on:
                detections, metadata = detect(drawing, image)
                x, y = self.get_coords()

                draw_hud(
                    drawing,
                    HudData(
                        detections=detections,
                        detect_metadata=metadata,
                        x=x,
                        y=y,
                        start_time=start_time,
                    ),
                )

            window_name = "camera_Feed"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, drawing)
            cv2.resizeWindow(window_name, 320, 240)
            key = cv2.waitKey(3)

            if key == ord("c"):
                self.get_logger().info("Saving image...")
                filename = f"Screenshot-{time()}.png"
                cv2.imwrite(filename, image)

            if key == ord("d"):
                self.detection_on = not self.detection_on
                self.get_logger().info(
                    f"Detection is now {'on' if self.detection_on else 'off'}"
                )

            if key == ord("r"):
                self.recording = not self.recording
                self.get_logger().info(
                    f"Recording is now {'on' if self.recording else 'off'}"
                )

            if key == ord("q"):
                self.get_logger().info("Shutting down...")
                out.release()
                rclpy.shutdown()
                return

            if self.recording:
                out.write(image)

            return
        except Exception as e:
            print(e)
            self.get_logger().error("Something seriously gone wrong, uhhhh ohhh")
            self.get_logger().error(f"{e}")
        finally:
            self.get_logger().info("Image processed")
            self.processing = False
        return


# Create a node of your class in the main and ensure it stays up and running
# handling exceptions and such
def main():

    from .cv_find_planets import load_ros_templates

    load_ros_templates()

    from ament_index_python.packages import get_package_share_directory
    import os

    package_path = get_package_share_directory("group_project")
    map_path = os.path.join(package_path, "worlds", "spacecraft_hard", "map")

    load_map(map_path)

    def signal_handler(sig, frame):
        print("Shutting down...")
        out.release()
        rclpy.shutdown()

    # Instantiate your class
    # And rclpy.init the entire node
    rclpy.init(args=None)
    feed = CameraFeed()

    # signal.signal(signal.SIGINT, signal_handler)
    # thread = threading.Thread(target=rclpy.spin, args=(feed,), daemon=True)
    # thread.start()

    rclpy.spin(feed)

    try:
        while rclpy.ok():
            continue
    except ROSInterruptException:
        pass

    # Remember to destroy all image windows before closing node
    cv2.destroyAllWindows()


# Check if the node is executing in the main path
if __name__ == "__main__":
    main()
