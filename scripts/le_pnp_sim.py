import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import pybullet as p
import pybullet_data as pd
import numpy as np
import m_pnp_env as panda_sim
import cv2
from util import *
from common_msgs.msg import ArmDataCollect

import time


class PandaSimNode(Node):

    def func_timer(func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            elapsed_time = time.time() - start_time
            self.get_logger().debug(
                f"{func.__name__} elapsed_time (s): {elapsed_time} running(hz): {1 / elapsed_time:.2f}"
            )
            return result

        return wrapper

    def __init__(self):
        super().__init__("panda_sim_node")
        self._init()

    def _init(self):
        self.bridge = CvBridge()

        # PyBullet初始化
        self._init_simulation()
        self.panda = panda_sim.PandaSim(p, [0, 0, 0])

        # 初始化定时器
        self._init_timer()
        # 注册参数回调函数
        self.declare_parameter("log_level", "debug")  # 默认值为 'info'
        self.add_on_set_parameters_callback(self._parameter_callback)
        self.use_model = True
        if self.use_model:
            from gr00t.eval.service import ExternalRobotInferenceClient

            try:
                print(f"RobotInferenceClient")
                # 先获取模型输入
                self.policy_client = ExternalRobotInferenceClient(host="localhost", port="5555")
            except Exception as e:
                print(f"simulation_step: {str(e)}")

    def _init_timer(self):

        # 创建仿真的定时器
        self.simulation_timer = self.create_timer(SIM_FREQ, self.simulation_step)

        # lerobot数据采集
        self.pub_le_robot_topic = False
        if self.pub_le_robot_topic:
            self.le_robot_data_pub = self.create_publisher(ArmDataCollect, "/le_robot_data", 10)
            self.le_robot_data_timer = self.create_timer(PUB_FREQ, self.le_robot_data_publish_step)

        if not self.pub_le_robot_topic:
            # 头部图像发布器
            self.top_image_pub = self.create_publisher(Image, "/pnp_sim/top_image_raw", 10)
            self.top_image_publish_timer = self.create_timer(PUB_FREQ, self.top_image_publish_step)
            # 关节发布器
            self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
            self.joint_states_publish_timer = self.create_timer(
                PUB_FREQ, self.joint_states_publish_step
            )
            # # 手臂相机 暂时关闭
            # self.hand_image_pub = self.create_publisher(Image, "/pnp_sim/hand_image_raw", 10)
            # self.hand_image_publish_timer = self.create_timer(
            #     PUB_FREQ, self.hand_image_publish_step
            # )
            # 左右相机
            self.left_image_pub = self.create_publisher(Image, "/pnp_sim/left_image_raw", 10)
            self.left_image_publish_timer = self.create_timer(
                PUB_FREQ, self.left_image_publish_step
            )
            self.right_image_pub = self.create_publisher(Image, "/pnp_sim/right_image_raw", 10)
            self.right_image_publish_timer = self.create_timer(
                PUB_FREQ, self.right_image_publish_step
            )

    def _parameter_callback(self, params):
        """参数回调函数"""
        log_level_map = {
            "debug": rclpy.logging.LoggingSeverity.DEBUG,
            "info": rclpy.logging.LoggingSeverity.INFO,
            "warn": rclpy.logging.LoggingSeverity.WARN,
            "error": rclpy.logging.LoggingSeverity.ERROR,
            "fatal": rclpy.logging.LoggingSeverity.FATAL,
        }

        for param in params:
            if param.name == "log_level":
                log_level = log_level_map.get(param.value)
                if log_level is not None:
                    set_logger_level(self.get_logger().name, log_level)
                else:
                    self.get_logger().warn(f"Unknown log level: {param.value}")
        return SetParametersResult(successful=True)

    def _init_simulation(self):
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setGravity(0, 0, -9.8)
        # 设置成20hz 苹果会直接掉下去
        p.setTimeStep(SIM_FREQ)

        # 设置一下相机的参数
        self._init_top_camera_para()
        self._init_left_camera_para()
        self._init_right_camera_para()
        # 相机内参
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=CAMERA_WIDTH / CAMERA_HEIGHT,
            nearVal=0.1,  # 调整近裁剪面
            farVal=2.0,  # 扩展远裁剪面
        )

    def _init_top_camera_para(self):
        # 设置俯视相机参数
        cam_eye = [0, 1.2, 1.2]  # 相机位置 (X,Y,Z)
        cam_target = [0, 0, 0]  # 注视场景中心

        cam_z_end = cam_target
        cam_y_end = np.array([0, 2.4, 0])

        cam_y_dir_in_world = cam_eye - cam_y_end
        # 相机图像的上下方向 投射到世界坐标系 对应的方向向量
        up_vector = cam_y_dir_in_world / np.linalg.norm(cam_y_dir_in_world)

        # cam_eye = [0.4, 0.4, 0.4]  # 相机位置 (X,Y,Z)
        # cam_target = [0.4, 0.4, 0]  # 注视场景中心
        # up_vector = [0, 1, 0]

        # 视图矩阵
        self.top_view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_eye,
            cameraTargetPosition=cam_target,
            cameraUpVector=up_vector,
        )
        # # 添加坐标轴可视化（验证对准效果）
        # [1,0,0] 是颜色
        # p.addUserDebugLine(cam_eye, cam_z_end, [0, 0, 1], 1)  # 红色视线
        # p.addUserDebugLine(cam_eye, cam_y_end, [0, 1, 0], 1)  # 绿色垂直线

    def _init_left_camera_para(self):
        cam_eye = [0.8, 0.8, 0.8]
        cam_target = [0.0, 0.0, 0.0]
        cam_y_end = np.array([1.6, 1.6, 0.0])
        cam_y_dir_in_world = cam_eye - cam_y_end
        up_vector = cam_y_dir_in_world / np.linalg.norm(cam_y_dir_in_world)
        # 视图矩阵
        self.left_view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_eye,
            cameraTargetPosition=cam_target,
            cameraUpVector=up_vector,
        )

    def _init_right_camera_para(self):
        cam_eye = [-0.8, 0.8, 0.8]
        cam_target = [0.0, 0.0, 0.0]
        cam_y_end = np.array([-1.6, 1.6, 0.0])
        cam_y_dir_in_world = cam_eye - cam_y_end
        up_vector = cam_y_dir_in_world / np.linalg.norm(cam_y_dir_in_world)
        # 视图矩阵
        self.right_view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_eye,
            cameraTargetPosition=cam_target,
            cameraUpVector=up_vector,
        )

    # @func_timer
    def simulation_step(self):
        if self.use_model:
            try:
                # 初始化动作缓冲区
                if not hasattr(self, "action_buffer"):
                    self.action_buffer = None
                    self.action_counter = 0
                    self.need_new_inference = True

                # 如果需要新的推理
                if self.need_new_inference:
                    # 收集观测数据
                    self.top_image_publish_step()
                    self.right_image_publish_step()
                    self.joint_states_publish_step()

                    # 构建观测字典
                    obs_dict = {
                        "video.right_view": self.right_image[np.newaxis, ...],
                        "video.head_view": self.top_image[np.newaxis, ...],
                        "state.right_arm": np.array(self.joints_angle[:7])[np.newaxis, :],
                        "state.right_gripper": np.array(self.joints_angle[7:])[np.newaxis, :],
                        "annotation.human.action.task_description": [
                            "Pick up the apple and place it on other box."
                        ],
                    }

                    # 执行推理，获取16帧动作
                    self.action_buffer = self.policy_client.get_action(obs_dict)
                    self.action_counter = 0
                    self.need_new_inference = False
                    print("New inference completed, starting execution...")

                # 执行当前帧动作
                if self.action_buffer is not None:
                    # 获取当前帧动作 (假设action_buffer是字典，包含多个动作序列)
                    current_action = {
                        k: v[self.action_counter] if v.ndim > 1 else v
                        for k, v in self.action_buffer.items()
                    }
                    combined_action = np.concatenate(
                        [current_action["action.right_arm"], current_action["action.right_gripper"]]
                    )
                    # 执行动作 (这里需要根据你的实际需求实现)
                    self.panda.step(self.use_model, combined_action)
                    p.stepSimulation()

                    # 更新计数器
                    self.action_counter += 1

                    # 检查是否需要重新推理
                    if self.action_counter >= 16:  # 假设每16帧重新推理一次
                        self.need_new_inference = True

            except Exception as e:
                print(f"simulation_step error: {str(e)}")
                self.need_new_inference = True  # 出错时强制重新推理
        else:
            self.panda.step()
            p.stepSimulation()

    # 使用装饰器
    @func_timer
    def top_image_publish_step(self):
        """俯视相机图像处理"""
        # 获取并转换图像
        _, _, rgb, _, _ = p.getCameraImage(
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            viewMatrix=self.top_view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        image = np.reshape(rgb, (CAMERA_HEIGHT, CAMERA_WIDTH, 4))[..., :3]
        self.top_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("top.jpg", image)
        self.top_ros_image = self.bridge.cv2_to_imgmsg(self.top_image, encoding="bgr8")
        if not self.pub_le_robot_topic:
            self.top_image_pub.publish(self.top_ros_image)

    @func_timer
    def left_image_publish_step(self):
        _, _, rgb, _, _ = p.getCameraImage(
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            viewMatrix=self.left_view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        image = np.reshape(rgb, (CAMERA_HEIGHT, CAMERA_WIDTH, 4))[..., :3]
        self.left_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("left.jpg", self.left_image)
        self.left_ros_image = self.bridge.cv2_to_imgmsg(self.left_image, encoding="bgr8")
        if not self.pub_le_robot_topic:
            self.left_image_pub.publish(self.left_ros_image)

    @func_timer
    def right_image_publish_step(self):
        _, _, rgb, _, _ = p.getCameraImage(
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            viewMatrix=self.right_view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        image = np.reshape(rgb, (CAMERA_HEIGHT, CAMERA_WIDTH, 4))[..., :3]
        self.right_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("right.jpg", self.right_image)
        self.right_ros_image = self.bridge.cv2_to_imgmsg(self.right_image, encoding="bgr8")
        if not self.pub_le_robot_topic:
            self.right_image_pub.publish(self.right_ros_image)

    # 使用装饰器
    @func_timer
    def hand_image_publish_step(self):
        # 获取末端执行器位姿
        ee_pos, ee_ori = self.panda.get_end_effector_pose()
        rot_matrix = np.array(p.getMatrixFromQuaternion(ee_ori)).reshape(3, 3)

        # 相机位置：在末端执行器位置的基础上稍微偏移
        cam_eye = ee_pos + rot_matrix @ np.array([0.0, 0.03, -0.035])  # 相机位置 (X,Y,Z)
        cam_target = ee_pos + rot_matrix @ np.array([0.0, 0.03, 1])  # 注视方向

        # 相机的上方向
        up_vector = np.linalg.inv(rot_matrix)[0, :3]  # @ np.array([0, -1, 0])  # 调整上方向为向下

        # 计算视图矩阵
        self.hand_view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_eye,
            cameraTargetPosition=cam_target,
            cameraUpVector=up_vector,
        )

        # 获取并转换图像
        _, _, rgb, _, _ = p.getCameraImage(
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            viewMatrix=self.hand_view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        image = np.reshape(rgb, (CAMERA_HEIGHT, CAMERA_WIDTH, 4))[..., :3]
        self.hand_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("hand.jpg", self.hand_image)

        # 如果需要发布图像到ROS话题，可以取消注释以下代码
        self.hand_ros_image = self.bridge.cv2_to_imgmsg(self.hand_image, encoding="bgr8")
        if not self.pub_le_robot_topic:
            self.hand_image_pub.publish(self.hand_ros_image)

    # 使用装饰器
    @func_timer
    def joint_states_publish_step(self):
        """Publish current joint states"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.position, msg.velocity = self.panda.get_current_joint_states()
        self.joints_angle = msg.position
        self.joint_name = msg.name
        if not self.pub_le_robot_topic:
            self.joint_state_pub.publish(msg)

    # 使用装饰器
    @func_timer
    def le_robot_data_publish_step(self):
        msg = ArmDataCollect()
        msg.header.stamp = self.get_clock().now().to_msg()

        self.top_image_publish_step()
        # self.left_image_publish_step()
        self.right_image_publish_step()
        # self.hand_image_publish_step()
        self.joint_states_publish_step()

        msg.top_image = self.top_ros_image
        # msg.left_image = self.left_ros_image
        msg.right_image = self.right_ros_image
        msg.joints_angle = self.joints_angle
        msg.joint_name = self.joint_name
        self.le_robot_data_pub.publish(msg)

    def destroy_node(self):
        p.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PandaSimNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
