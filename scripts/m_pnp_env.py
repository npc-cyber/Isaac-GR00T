# m_pnp_env.py
import time
import numpy as np
import math
import scipy.spatial.transform as st
from scipy.interpolate import CubicSpline
from util import *

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11  # 8
pandaNumDofs = 7

ll = [-7] * pandaNumDofs
# upper limits for null space (todo: set them to proper range)
ul = [7] * pandaNumDofs
# joint ranges for null space (todo: set them to proper range)
jr = [7] * pandaNumDofs
# restposes for null space
# 最后两个值对应夹爪的 2个平移关节 爪子有两个指 不是所以需要控制两个指的位置
# [0.0, 0.0] 完全闭合
# [0.1, 0.0] 有一个指不在中间位置
panda_arm_joint_positions = np.array([90.0, 15.0, 0.0, -90.0, 0.0, 135.0, 45.0]) * np.pi / 180
panda_hand_joint_positions = np.array([0.07, 0.07])
# 两个合成一个np
jointPositions = np.concatenate([panda_arm_joint_positions, panda_hand_joint_positions])
rp = jointPositions
# 创建一个正对的姿态四元数
zero_quat = np.array([0.0, 0.0, 0.0, 1.0])


class PandaSim(object):
    def __init__(self, bullet_client, offset):
        self.bullet_client = bullet_client
        self.offset = np.array(offset)
        self.apples = []
        self.gripped_apple = None
        self.middle_robot_pos = np.array([-0.45, 0.45, 0.3])
        self.target_robot_pos = np.array([0.45, 0.45, 0.3]) + self.offset
        self.current_target = None
        self.grasp_offset = 0.025  # 抓取高度偏移
        self.current_target_joint_pos = None
        self.position_threshold = 0.025  # 位置到达阈值
        self.gripper_state = np.array([1.0, 1.0, 0.0])  # 状态改成这个也许好训练一些
        self.t = 0.0

        # 新增状态控制参数
        self.state = 0  # 0:空闲 1:移动至目标 2:抓取中 3:运输中
        self.control_dt = SIM_FREQ  # 控制周期

        # 初始化环境
        self.prepare_sim_env()
        self.prepare_sim_robot()

    def reset(self):
        pass

    def prepare_sim_env(self):
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        # 加载盒子
        box_positions = [[0.45, 0.45, 0.015], [-0.45, 0.45, 0.015]]
        for pos in box_positions:
            self.bullet_client.loadURDF(
                "tray/traybox.urdf", np.array(pos) + self.offset, zero_quat, flags=flags
            )

        # 加载苹果
        obj_path = "/home/yangrui/trilib/groot/pybullet/pybullet-URDF-models/urdf_models/models/plastic_apple/model.urdf"
        apple_z_positions = [0.02]
        for z in apple_z_positions:
            apple_id = self.bullet_client.loadURDF(
                obj_path, [-0.47, 0.47, z] + self.offset, flags=flags
            )
            self.apples.append({"id": apple_id, "is_gripped": False, "is_in_target": False})

    def prepare_sim_robot(self):
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        # 加载机械臂
        self.panda = self.bullet_client.loadURDF(
            "franka_panda/panda.urdf",
            np.array([0, 0, 0]) + self.offset,
            np.array([0, 0, 0, 1]),
            useFixedBase=True,
            flags=flags,
        )

        # 初始化关节状态
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.panda)):

            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)

            jointName = info[1]
            jointType = info[2]
            if jointType == self.bullet_client.JOINT_PRISMATIC:
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
                # print(" j ",j," index ",index)
            if jointType == self.bullet_client.JOINT_REVOLUTE:
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
                # print(" j ",j," index ",index)

        # 获取初始末端位置
        self.ee_home_pos, self.ee_home_orn = self.get_end_effector_pose()
        grip_pose = np.eye(3)
        grip_pose[1, 1] = -1
        grip_pose[2, 2] = -1
        self.ee_home_orn = st.Rotation.from_matrix(grip_pose).as_quat()

    def control_gripper(self, finger_target):
        """简化版夹爪控制"""
        self.bullet_client.setJointMotorControl2(
            self.panda, 9, self.bullet_client.POSITION_CONTROL, finger_target, force=20
        )
        self.bullet_client.setJointMotorControl2(
            self.panda, 10, self.bullet_client.POSITION_CONTROL, finger_target, force=20
        )
        # # 同时设置两个夹爪关节
        # self.bullet_client.setJointMotorControlArray(
        #     self.panda,
        #     [9, 10],  # 夹爪关节索引
        #     self.bullet_client.POSITION_CONTROL,
        #     targetPositions=[finger_target] * 2,  # 两个关节相同目标值
        #     forces=[15] * 2,  # 增加控制力
        #     positionGains=[0.8] * 2,  # 添加位置增益
        # )

    def find_reachable_apple(self):
        """寻找可到达的苹果"""
        for apple in self.apples:
            if apple["is_in_target"] or apple["is_gripped"]:
                continue

            # 获取苹果当前位置
            apple_pos, _ = self.bullet_client.getBasePositionAndOrientation(apple["id"])
            target_pos = np.array(apple_pos) + [0, 0, self.grasp_offset]

            # 计算逆运动学
            joint_pos = self.calculate_ik(target_pos, self.ee_home_orn)
            if self.validate_ik(joint_pos, target_pos):
                return apple, target_pos, joint_pos
        return None, None, None

    def set_target_pos(self, target_pos):
        # 转换为numpy数组确保类型一致
        target_pos = np.asarray(target_pos)
        # 使用numpy的数组比较方法
        if self.current_target_joint_pos is not None and np.allclose(
            self.current_target_joint_pos, target_pos, atol=1e-3
        ):
            return
        print("change target pos")
        self.current_target_joint_pos = target_pos
        self.target_joint_pos = self.calculate_ik(self.current_target_joint_pos, self.ee_home_orn)

    def move_to_target(self):
        """移动到目标位置"""
        current_pos, _ = self.get_end_effector_pose()
        if self.target_joint_pos is None:
            return False
        cur_position_error = np.linalg.norm(current_pos - self.current_target_joint_pos)
        if cur_position_error > self.position_threshold:
            self.apply_joint_positions(self.target_joint_pos)
            return False
        return True

    def calculate_ik(self, target_pos, ee_home_orn):
        """计算逆运动学"""
        return self.bullet_client.calculateInverseKinematics(
            self.panda,
            pandaEndEffectorIndex,
            target_pos,
            ee_home_orn,
            ll,
            ul,
            jr,
            rp,
            maxNumIterations=50,  # 调整为平衡速度与精度
        )

    def validate_ik(self, joint_pos, target_pos):
        """验证逆解有效性"""
        # 检查关节限位
        for i in range(pandaNumDofs):
            if not (ll[i] <= joint_pos[i] <= ul[i]):
                return False
        return True

    def apply_joint_positions(self, target_joint_pos):
        """应用关节位置控制（改进版轨迹生成）"""
        # 应用每个关节的位置控制
        for j in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(
                self.panda,
                j,
                self.bullet_client.POSITION_CONTROL,
                targetPosition=target_joint_pos[j],
                force=5 * 240.0,
                maxVelocity=0.9,
            )

    def grip_apple(self, apple):
        """抓取苹果（添加夹爪闭合控制）"""
        # 闭合夹爪
        self.gripper_state = np.array([0.0, 0.0])
        self.control_gripper(finger_target=0.01)
        apple["is_gripped"] = True
        self.gripped_apple = apple

    def release_apple(self):
        """释放苹果到目标位置（添加夹爪张开控制）"""
        if not self.gripped_apple:
            return
        self.gripper_state = np.array([1.0, 1.0])
        # 张开夹爪
        self.control_gripper(finger_target=0.08)
        # 更新状态
        self.gripped_apple["is_gripped"] = False
        self.gripped_apple["is_in_target"] = True
        self.gripped_apple = None

    def step(self, use_model=False, joint_pos=None):
        """主控制循环（添加状态机逻辑）"""
        self.t += self.control_dt
        if use_model:
            self.apply_joint_positions(joint_pos)
            finger_target = 1.0 if (joint_pos[7] + joint_pos[8]) / 2 > 0.9 else 0.0
            # print(finger_target)
            # if finger_target == 1.0:
            #     self.gripper_state[2] = self.gripper_state[2] + 1.0
            # if finger_target == 1.0 and self.gripper_state[0] == 0.0:
            #     finger_target = 1.0
            self.gripper_state[0] = finger_target
            self.gripper_state[1] = finger_target
            self.control_gripper(0.045 if finger_target == 1.0 else 0.01)
        else:
            self.not_use_model_func()

    def not_use_model_func(self):
        remaining = [a for a in self.apples if not a["is_in_target"]]
        if not remaining:
            print("All apples moved!")
            return
        current_pos, _ = self.get_end_effector_pose()

        # 状态机逻辑
        if self.state == 0:  # 寻找目标
            apple, target_pos, joint_pos = self.find_reachable_apple()
            if apple:
                self.current_target = (apple, target_pos, joint_pos)
                self.state = 1
        elif self.state == 1:  # 移动至目标上方
            print("移动至目标上方")
            self.set_target_pos(self.middle_robot_pos)
            if self.move_to_target():
                self.state = 2
        elif self.state == 2:  # 移动到目标位置
            print("移动到目标位置")
            _, target_pos, joint_pos = self.current_target
            self.set_target_pos(target_pos)
            if self.move_to_target():
                self.state = 3
        elif self.state == 3:  # 执行抓取
            print("执行抓取")
            apple, _, _ = self.current_target
            self.grip_apple(apple)
            self.state = 4  # 进入运输状态
        elif self.state == 4:
            print("transport_apple")
            try:
                self.set_target_pos(self.target_robot_pos)
                if self.move_to_target():
                    self.release_apple()
                    self.state = 0  # 重置状态
            except Exception as e:
                print(f"运输过程中发生错误: {str(e)}")

    def get_current_joint_states(self):
        """获取当前关节状态"""
        joint_names = []
        joint_positions = []
        joint_velocities = []
        for joint_index in range(self.bullet_client.getNumJoints(self.panda)):
            info = self.bullet_client.getJointInfo(self.panda, joint_index)
            jointName = info[1]
            jointType = info[2]
            if (
                jointType == self.bullet_client.JOINT_PRISMATIC
                or jointType == self.bullet_client.JOINT_REVOLUTE
            ):
                js = self.bullet_client.getJointState(self.panda, joint_index)
                joint_positions.append(js[0])
                joint_velocities.append(js[1])
                joint_names.append(jointName.decode("utf-8"))
        # 我之前为什么要这样设置夹爪 主要是夹爪非常难训练
        # 我设置的数据是0.01 采集数据的时候夹爪到0.04就夹住了
        # 这样的话 似乎改action也行
        # 不是很显著
        joint_positions[7] = self.gripper_state[0]
        joint_positions[8] = self.gripper_state[1]
        return joint_names, joint_positions, joint_velocities

    def get_end_effector_pose(self):
        """获取末端执行器位姿"""
        link_state = self.bullet_client.getLinkState(
            self.panda, pandaEndEffectorIndex, computeForwardKinematics=1
        )
        return link_state[4], link_state[5]
