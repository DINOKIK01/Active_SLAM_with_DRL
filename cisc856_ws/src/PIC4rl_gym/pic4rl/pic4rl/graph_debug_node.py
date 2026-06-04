#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

import numpy as np

from pic4rl.utils.graph_utils import GraphMap, update_graph_from_lidar


class GraphDebugNode(Node):
    def __init__(self):
        super().__init__("graph_debug_node")

        # -----------------------------
        # Graph initialisieren
        # -----------------------------
        self.graph = GraphMap(width=6, height=6)

        # -----------------------------
        # Subscriber
        # -----------------------------
        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )

        # -----------------------------
        # Datenpuffer
        # -----------------------------
        self.robot_pose = None
        self.lidar_data = None

        # Timer für Updates
        self.timer = self.create_timer(1.5, self.update)

        self.get_logger().info("Graph Debug Node gestartet")

    # -----------------------------
    # Callbacks
    # -----------------------------
    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Quaternion → Yaw
        q = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

        self.robot_pose = [x, y, yaw]

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)

        # Clip auf 10m (wie du willst)
        ranges = np.clip(ranges, 0.0, 10.0)
        scan_range = np.minimum.reduceat(ranges, np.arange(0, len(ranges), 10))

        self.lidar_data = scan_range

    # -----------------------------
    # Update Loop
    # -----------------------------
    def update(self):
        if self.robot_pose is None or self.lidar_data is None:
            return

        update_graph_from_lidar(
            graph=self.graph,
            robot_pose=self.robot_pose,
            lidar_measurements=self.lidar_data,
        )

        self.graph.print_graph(self.robot_pose)
        print("\n----------------------------\n")

    # -----------------------------
    # Debug Ausgabe
    # -----------------------------
    def print_debug(self):
        node = self.get_current_node()

        if node is None:
            return

        self.get_logger().info(
            f"Node ({node.x},{node.y}) | visited={node.visited} | visits={node.visit_count}"
        )

        self.get_logger().info(f"Edges: {node.edges}")

    def get_current_node(self):
        if self.robot_pose is None:
            return None

        from pic4rl.utils.env_utils import get_coordinates
        i, j = get_coordinates(self.robot_pose)

        return self.graph.get_node(i, j)

    # -----------------------------
    # Helper
    # -----------------------------
    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)

    node = GraphDebugNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()