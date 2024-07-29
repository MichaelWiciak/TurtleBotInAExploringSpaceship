# modified from lab 4

from group_project import coordinates
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from math import sin, cos
import threading

import signal


class ToPoint(Node):
    def __init__(self):
        super().__init__('navigation_goal_action_client')

        self.declare_parameter('coordinates_file_path', '')
        coordinates_file_path = self.get_parameter('coordinates_file_path').get_parameter_value().string_value
        self.coordinates = coordinates.get_module_coordinates(coordinates_file_path)

        self.declare_parameter('module', '')
        self.module = self.get_parameter('module').get_parameter_value().string_value
        self.declare_parameter('place', '')
        self.place = self.get_parameter('place').get_parameter_value().string_value

        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def send_goal(self, x, y, yaw):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # Orientation
        goal_msg.pose.pose.orientation.z = sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2)

        self.action_client.wait_for_server()
        self.send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # NOTE: if you want, you can use the feedback while the robot is moving.

def main(args=None):
    rclpy.init(args=args)

    def signal_handler(sig, frame):
            rclpy.shutdown()

    # Instantiate your class
    # And rclpy.init the entire node
    to_point = ToPoint()

    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(to_point,), daemon=True)
    thread.start()


    to_point = ToPoint()
    print(to_point.coordinates)
    print(to_point.module)
    print(to_point.place)

    module = to_point.coordinates.module_1 if to_point.module == '1' else to_point.coordinates.module_2
    place = module.entrance if to_point.place == 'entrance' else module.center

    to_point.send_goal(place.x, place.y, 0)  # example coordinates
    rclpy.spin(to_point)

if __name__ == '__main__':
    main()
