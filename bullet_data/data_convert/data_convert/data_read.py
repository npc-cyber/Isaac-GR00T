import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import message_filters
from pathlib import Path
import os.path as osp
import shutil
from common_msgs.msg import ArmDataCollect

class InsertPort(Node):
    def __init__(self):
        super().__init__('inserport_node')
        self.declare_parameter('debug_mode', False)
        self.declare_parameter("detection_freq", 10)
    
        self.right_camera_info = self.create_subscription(
            ArmDataCollect,
            "/right_arm_data_cmd",
            self.camera_info_callback_right,
            10,
        )

        # self.subscription_rgb = message_filters.Subscriber(
        #     self,
        #     ArmDataCollect,
        #     '/right_arm_data_cmd')
        # self.subscription_depth = message_filters.Subscriber(
        #     self,
        #     Image,
        #     '/debug/yolov11s_segmentation')
        # self.subscription_info = message_filters.Subscriber(
        #     self,
        #     CameraInfo,
        #     '/perception/camera_info')
 
        self.bridge = CvBridge()
        
        # self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
        #     [self.subscription_rgb],  #, self.subscription_depth, self.subscription_info
        #     10,0.1)

        
        # self.time_synchronizer.registerCallback(self.insertPort_callback)
        
    
                 
    def camera_info_callback_right(self, msg): #, depth_msg, info_msg
        try:
            self.get_logger().info("Flatness check left: {}".format(msg.arm_name))
            # self.image = rgb_msg
            # self.cameraId = 1
            img = self.bridge.imgmsg_to_cv2( msg.right_image)
            # # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            # right = msg.right_image
            # depth_image = self.bridge.imgmsg_to_cv2(depth_msg)
            # depth =  depth_image
            # print(depth.shape, depth[163, 232])
            cv2.imwrite("./test/" + str(timestamp) + ".png", img)
            # cv2.imwrite("./test/" + str(timestamp) + "_seg.png", depth)
            
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")
            



        
def main(args=None):
    rclpy.init(args=args)
    node = InsertPort()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()