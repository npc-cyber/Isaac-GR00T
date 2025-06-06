#!/usr/bin/env python3
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rclpy
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from common_msgs.msg import ArmDataCollect
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
import cv2
import numpy as np
import pandas as pd
import ffmpeg
from cv_bridge import CvBridge
from tqdm import tqdm

class Bag2LeRobot:
    """将ROS bag数据转换为LeRobot格式的数据集"""

    def __init__(
        self,
        bag_path: str,
        output_root: str,
        task_language: str,
        sync_topic: str = "/synchronized_data",
    ):
        """
        初始化转换器

        Args:
            bag_path: ROS bag文件路径
            output_root: 输出目录
            sync_topic: 同步数据的topic名称
        """
        self.bag_path = bag_path
        self.output_root = Path(output_root)
        self.sync_topic = sync_topic
        self.bridge = CvBridge()
        # self.batch_mode = batch_mode
        self.task_language = task_language

        # 创建日志记录器
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        

    def convert(self):
        """执行转换过程"""
        try:
            # if self.batch_mode:
            self._create_directory_structure()
            self._batch_convert()
            # else:
            #     self._single_bag_convert(self.bag_path)

        except Exception as e:
            self.logger.error(f"转换过程出错: {str(e)}")
            raise

    # def _single_bag_convert(self, bag_path):
    #     """转换单个bag文件"""
    #     try:
    #         self.bag_path = bag_path
    #         # 1. 创建目录结构
    #         self._create_directory_structure()

    #         # 2. 读取并处理bag数据
    #         data, data_info = self.read_rosbag_data()
    #         if not data:
    #             self.logger.error(f"未能从bag {bag_path.name} 中读取到有效数据")
    #             return
            
    #         # 3. 生成数据集
    #         self._generate_dataset(data, data_info)
    #         self.logger.info(f"数据集 {bag_path} 转换完成")

    #     except Exception as e:
    #         self.logger.error(f"转换 {bag_path} 过程出错: {str(e)}")
    #         raise
    def _create_directory_structure(self):
        """创建LeRobot格式所需的目录结构"""
        # 创建基本目录
        (self.output_root / "meta").mkdir(parents=True, exist_ok=True)
        (self.output_root / "data/chunk-000").mkdir(parents=True, exist_ok=True)

        # 为每个相机视角创建视频目录
        for camera in ['right', 'head']:  # 'left'
            (self.output_root / f"videos/chunk-000/observation.images.{camera}_view").mkdir(parents=True, exist_ok=True)

    def _batch_convert(self):
        """批量转换多个bag文件"""
        # 检查bag_path是否是目录
        self.bag_path = Path(self.bag_path)
        if not self.bag_path.is_dir():
            self.logger.error(f"{self.bag_path} 不是一个有效的目录")
            return

        # 查找所有可能的bag目录（包含metadata.yaml和.db3文件的目录）
        bag_dirs = []

        # 递归查找所有包含 metadata.yaml 的目录
        for root_dir in self.bag_path.rglob("metadata.yaml"):
            # 获取 metadata.yaml 所在目录
            parent_dir = root_dir.parent
            # 检查该目录下是否存在 .db3 文件
            db3_files = list(parent_dir.glob("*.db3"))
            if db3_files:
                # 将 .db3 文件路径添加到结果列表
                bag_dirs.extend([db3_file for db3_file in db3_files])

        if not bag_dirs:
            self.logger.error(f"在 {self.bag_path} 中未找到有效的ROS bag目录")
            return

        self.logger.info(f"找到 {len(bag_dirs)} 个bag目录, 开始批量转换")

        # 维护全局索引
        global_episode_idx = 0
        global_frame_idx = 0
        total_frames = 0
        all_episodes_meta = []

        # 首先计算所有bag的总帧数
        for bag_dir in bag_dirs:
            data, _ = self.read_rosbag_data(str(bag_dir))
            if data:
                total_frames += len(data) - 1
        
        # 处理每个bag
        for i, bag_dir in enumerate(bag_dirs):
            self.logger.info(f"[{i+1}/{len(bag_dirs)}] 开始处理 {bag_dir}")
            
            # 读取当前bag的数据
            data, robot_info = self.read_rosbag_data(str(bag_dir))
            if not data:
                continue

            # 生成数据集，传入全局索引
            episodes_meta, fps = self._generate_dataset(
                data, 
                robot_info,
                global_episode_idx=global_episode_idx,
                global_frame_idx=global_frame_idx
            )
            
           # 更新全局索引
            global_episode_idx += len(episodes_meta)
            global_frame_idx += len(data) - 1
                
            # 收集所有episodes的元数据
            all_episodes_meta.extend(episodes_meta)

       # 最后生成全局的元数据文件
        self._generate_metadata_files(
            num_episodes=global_episode_idx,
            robot_info=robot_info,  # 使用最后一个bag的robot_info
            total_frames=total_frames,
            episodes_meta=all_episodes_meta,
            fps=fps
        )
        
    def _create_directory_structure(self):
        """创建LeRobot格式所需的目录结构"""
        # 创建基本目录
        (self.output_root / "meta").mkdir(parents=True, exist_ok=True)
        (self.output_root / "data/chunk-000").mkdir(parents=True, exist_ok=True)

        # 为每个相机视角创建视频目录
        for camera in ['right', 'head']:  # 'left'
            (self.output_root / f"videos/chunk-000/observation.images.{camera}_view").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_rosbag_options(path, serialization_format='cdr'):
        storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')

        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=serialization_format,
            output_serialization_format=serialization_format)

        return storage_options, converter_options
    
    
    def read_rosbag_data(self, bag_path):
        """读取bag数据并返回处理后的数据列表"""
        data = []
        robot_info = None
        try:
            storage_options, converter_options = self.get_rosbag_options(bag_path)

            reader = rosbag2_py.SequentialReader()
            reader.open(storage_options, converter_options)

            topic_types = reader.get_all_topics_and_types()
            type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

            while reader.has_next():
                (topic, data_msg, t) = reader.read_next()
                if topic == '/le_robot_data':
                    msg_type = type_map[topic]
                    if msg_type == 'common_msgs/msg/ArmDataCollect':
                        msg = deserialize_message(data_msg, ArmDataCollect)
                        # # 解析头部信息
                        # header: Header = msg.header
                        # print(f"Header: stamp={header.stamp}, frame_id={header.frame_id}")

                        # # 解析机械臂名称
                        # arm_name = msg.arm_name
                        # print(f"Arm name: {arm_name}")

                        # # 解析笛卡尔空间位置和姿态
                        # pose: PoseStamped = msg.pose
                        # position = pose.pose.position
                        # orientation = pose.pose.orientation
                        # print(f"Pose: position=(x={position.x}, y={position.y}, z={position.z})")
                        # print(f"Orientation: (x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w})")

                        # # 解析旋转类型
                        # rotation_type = msg.rotation_type
                        # print(f"Rotation type: {rotation_type}")

                        # # 解析关节角信息
                        # joint_names = msg.joint_name
                        # joint_angles = msg.joints_angle
                        # for i in range(len(joint_names)):
                        #     print(f"Joint {joint_names[i]} angle: {joint_angles[i]}")

                        # # 解析夹爪开合度
                        # gripper_distance = msg.gripper_distance
                        # print(f"Gripper distance: {gripper_distance}")

                        # # 解析图像信息
                        # # 这里只是简单打印图像的高度和宽度，你可以根据需求进一步处理图像数据
                        # right_image: Image = msg.right_image
                        # top_image: Image = msg.top_image
                        # print(f"Right image: height={right_image.height}, width={right_image.width}")
                        # print(f"Top image: height={top_image.height}, width={top_image.width}")
                       
                       # 第一条消息时获取机器人信息
                        if robot_info is None:
                            robot_info = {
                                # "left_arm": {
                                #     "joint_names": msg.joint_name
                                # },
                                "right_arm": {
                                    "joint_names": msg.joint_name
                                }
                            }

                        # 处理同步数据
                        processed_data = self._process_synchronized_data(msg)
                        if processed_data:
                            data.append(processed_data)

                # self.logger.info(f"成功读取 {len(data)} 条同步数据记录")
        except Exception as e:
            self.logger.error(f"读取bag文件失败: {str(e)}")
            raise

        return data, robot_info
   

    def _process_synchronized_data(self, msg):
        """处理同步数据消息"""
        try:
            # # 确保所有时间戳一致
            # msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            # if abs(msg_timestamp - timestamp) > 1e-6:  # 允许1微秒的误差
            #     self.logger.warning(f"时间戳不匹配: bag={timestamp}, msg={msg_timestamp}")

            # 转换图像数据
            # left_img = self.bridge.imgmsg_to_cv2(msg.left_image, "bgr8")
            right_img = self.bridge.imgmsg_to_cv2(msg.right_image, "bgr8")
            top_image = self.bridge.imgmsg_to_cv2(msg.top_image, "bgr8")  # 注意这里改成了head_img
            

            images = {
                # 'left_view': left_img,
                'right_view': right_img,
                'head_view': top_image
            }

            # 处理机械臂数据
            def get_arm_state(arm_data):
                return msg.joints_angle
            
            # 获取左右臂状态
            # left_state = get_arm_state(msg.left_arm_data)
            right_state = get_arm_state(msg)

            # 合并状态向量
            robot_state = np.concatenate([right_state]) #left_state, 
            robot_state = np.array(right_state)
            return {
                'fps': 15,  # 按实际修改
                'observation': {
                    'state': robot_state,
                },
                'images': images
            }

        except Exception as e:
            self.logger.error(f"处理同步数据失败: {str(e)}")
            return None

  
    def _generate_dataset(self, data, robot_info, global_episode_idx=0, global_frame_idx=0):
        """生成LeRobot格式数据集"""
        episodes_meta = []
        
        # 确保所有数据长度一致，都只取到倒数第二帧
        data_length = len(data) - 1  
        
        # 使用全局frame_idx而不是局部索引
        current_frame_idx = global_frame_idx
        
        # 生成当前episode的索引序列
        current_indices = np.arange(current_frame_idx, current_frame_idx + data_length, dtype=np.int64)

        # 生成时间戳序列
        frame_interval = 1.0 / data[0]['fps']
        base_timestamp = 0
        timestamps = np.array([base_timestamp + i * frame_interval for i in range(data_length)])

        # 生成parquet文件数据
        episode_dict = {
            # 状态数据，取到倒数第二帧
            "observation.state": [frame['observation']['state'] for frame in data[:-1]] if 'observation' in data[0] else np.zeros((data_length, 7), dtype=np.float64).tolist(),
            # 动作数据，使用从第二帧开始到最后一帧的状态
            "action": [frame['observation']['state'] for frame in data[1:]] if 'observation' in data[0] else np.zeros((data_length, 7), dtype=np.float64).tolist(),
            # 时间戳和其他元数据
            "timestamp": timestamps,
            "episode_index": np.full(data_length, global_episode_idx, dtype=np.int64),
            "index": current_indices,
            "task_index": np.zeros(data_length, dtype=np.int64),
            "annotation.human.action.task_description": np.zeros(data_length, dtype=np.int64),
            "annotation.human.validity": np.ones(data_length, dtype=np.int64),
            "next.reward": np.zeros(data_length, dtype=np.float64),
            "next.done": np.array([False] * (data_length - 1) + [True], dtype=bool)
        }

        # 保存parquet文件
        df = pd.DataFrame(episode_dict)
        df.to_parquet(self.output_root / f"data/chunk-000/episode_{global_episode_idx:06d}.parquet")

        # 保存视频文件
        for view_name in data[0]['images'].keys():
            # 收集该视图下的所有帧，只取到倒数第二帧
            all_frames = []
            for frame_data in data[:-1]:
                all_frames.append(frame_data['images'][view_name])
            
            # 保存视频序列
            self._save_video_data(all_frames, global_episode_idx, view_name)

        # 收集episode元数据
        episode_meta = {
            "episode_index": global_episode_idx,
            "tasks": [self.task_language, "valid"],
            "length": data_length
        }
        episodes_meta.append(episode_meta)

        return episodes_meta, data[0]['fps']
    
    def _save_video_data(self, frames, episode_idx, view_name, fps=15.0) -> None:
        """保存视频数据为mp4格式"""
        video_path = self.output_root / f"videos/chunk-000/observation.images.{view_name}/episode_{episode_idx:06d}.mp4"
        output_dir = Path(video_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        # if view_name == "head_view":
        #     h, w = 360, 640
        # elif view_name == "right_view" or view_name == "left_view":
        #     h, w = 480, 848
        # else:
        #     h, w = 256, 256
        h, w = 256, 256
        # 确保frames是列表
        if not isinstance(frames, list):
            frames = [frames]

        # 确保帧是uint8类型
        if len(frames) > 0 and hasattr(frames[0], 'dtype') and frames[0].dtype != np.uint8:
            frames = [np.clip(frame * 255, 0, 255).astype(np.uint8) for frame in frames]

        # # 统一调整尺寸为256x256
        frames = [cv2.resize(frame, (h, w), interpolation=cv2.INTER_LANCZOS4) for frame in frames]

       
        # 创建 FFmpeg 进程
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{w}x{h}', r=fps)
            .output(str(video_path), vcodec='libx264', pix_fmt='yuv420p', crf=0)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        try:
            for frame in frames:
                # 转换颜色空间为 BGR（FFmpeg 默认使用 BGR）
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                process.stdin.write(frame_bgr.astype(np.uint8).tobytes())

        finally:
            # 关闭进程
            process.stdin.close()
            process.wait()

    # def _save_video_data(self, frames, episode_idx, view_name) -> None:
    #     """保存视频数据为mp4格式"""
    #     video_path = self.output_root / f"videos/chunk-000/observation.images.{view_name}/episode_{episode_idx:06d}.mp4"
        
    #     # if view_name == "head_view":
    #     #     h, w = 360, 640
    #     # elif view_name == "right_view" or view_name == "left_view":
    #     #     h, w = 480, 848
    #     # else:
    #     #     h, w = 256, 256
    #     h, w = 256, 256
    #     # 确保frames是列表
    #     if not isinstance(frames, list):
    #         frames = [frames]

    #     # 确保帧是uint8类型
    #     if len(frames) > 0 and hasattr(frames[0], 'dtype') and frames[0].dtype != np.uint8:
    #         frames = [np.clip(frame * 255, 0, 255).astype(np.uint8) for frame in frames]

    #     # # 统一调整尺寸为256x256
    #     frames = [cv2.resize(frame, (h, w), interpolation=cv2.INTER_LANCZOS4) for frame in frames]

    #     # 创建VideoWriter对象
    #     # 直接使用整数值代表编码器
    #     # fourcc = 0x7634706d  # 'mp4v' 的整数表示
    #     # out = cv2.VideoWriter(str(video_path), fourcc, 15.0, (w, h))

    #     # 选择高画质编码器（H.264）
    #     fourcc = cv2.VideoWriter_fourcc(*'H264')  # 或 'XVID' 用于 AVI
    #     out = cv2.VideoWriter(str(video_path), fourcc, 15.0, (w, h))

    #     try:
    #         # 写入帧
    #         for frame in frames:
    #             out.write(frame)
    #     finally:
    #         out.release()

    def _generate_metadata_files(self, num_episodes, robot_info, total_frames, episodes_meta, fps):
        """生成metadata文件"""
        # 1. modality.json
        modality_meta = {
            "state": {
                "right_arm": {
                    "start": 0,
                    "end": 7,
                },
                "right_gripper": {
                    "start": 7,
                    "end": 9
                }
            },
            "action": {
                "right_arm": {
                    "start": 0,
                    "end": 7
                },
                "right_gripper": {
                    "start": 7,
                    "end": 9
                }
            },
            "video": {
                "right_view": {
                    "original_key": "observation.images.right_view"
                },
                "head_view": {
                    "original_key": "observation.images.head_view"
                }
            },
            "annotation": {
                "human.action.task_description": {},
                "human.validity": {}
            }
        }

        with open(self.output_root / "meta/modality.json", "w") as f:
            json.dump(modality_meta, f, indent=2)

        # 2. episodes.jsonl
        with open(self.output_root / "meta/episodes.jsonl", "w") as f:
            for episode in episodes_meta:
                f.write(json.dumps(episode) + "\n")

        # 3. info.json
        # Assemble metadata v2.0
        state_names = []
        # 添加左臂关节名称
        # for joint in robot_info["left_arm"]["joint_names"]:
        #     state_names.append(f"left_{joint}")
        # state_names.append("left_gripper_distance")

        # 添加右臂关节名称
        for joint in robot_info["right_arm"]["joint_names"]:
            state_names.append(f"{joint}")
        # state_names.append("right_gripper_distance")

        
        info_meta = {
            "codebase_version": "v2.0",
            "robot_type": "Panda",  # 改为Panda
            "total_episodes": num_episodes,
            "total_frames": total_frames,
            "total_tasks": 1, # num_episodes,  # 每个episode一个任务
            "total_videos": 2 * num_episodes,  # left_view, right_view, head_view
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": fps,
            "splits": {
                "train": "0:100"
            },
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                # "observation.images.left_view": {
                #     "dtype": "video",
                #     "shape": [256, 256, 3],
                #     "names": ["height", "width", "channel"],
                #     "video_info": {
                #         "video.fps": fps,
                #         "video.codec": "h264",
                #         "video.pix_fmt": "yuv420p",
                #         "video.is_depth_map": False,
                #         "has_audio": False
                #     }
                # },
                "observation.images.right_view": {
                    "dtype": "video",
                    "shape": [256, 256, 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": fps,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                },
                "observation.images.head_view": {
                    "dtype": "video",
                    "shape": [256, 256, 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": fps,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                },
                "observation.state": {
                    "dtype": "float64",
                    "shape": [len(state_names)],  # 6(左臂关节) + 1(左夹爪) + 6(右臂关节) + 1(右夹爪)
                    "names": state_names
                },
                "action": {
                    "dtype": "float64",
                    "shape": [len(state_names)],  # 6(左臂关节) + 1(左夹爪) + 6(右臂关节) + 1(右夹爪)
                    "names": state_names
                },
                "timestamp": {
                    "dtype": "float64",
                    "shape": [1]
                },
                "annotation.human.action.task_description": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "annotation.human.validity": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "episode_index": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "index": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "next.reward": {
                    "dtype": "float64",
                    "shape": [1]
                },
                "next.done": {
                    "dtype": "bool",
                    "shape": [1]
                }
            }
        }

        with open(self.output_root / "meta/info.json", "w") as f:
            json.dump(info_meta, f, indent=2)


        # 4. tasks.jsonl
        # for i in range(num_episodes):
        tasks_meta = [{
                "task_index": 0,
                "task": self.task_language
            },
            {
                "task_index": 1,
                "task": "valid"
            }]

        with open(self.output_root / "meta/tasks.jsonl", "w") as f:
            for task in tasks_meta:
                f.write(json.dumps(task) + "\n")

def main():
    # 命令行参数解析
    import argparse
    parser = argparse.ArgumentParser(description='Convert ROS bag to LeRobot format')
    parser.add_argument('--bag_path', type=str, required=True, help='Path to the ROS bag file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the dataset')
    parser.add_argument('--sync_topic', type=str, default='/le_robot_data', help='Synchronized data topic name')
    parser.add_argument('--language', type=str, default='Pick up the apple and place it on other box.', help='Specify the natural language description of the task.')

    args = parser.parse_args()

    try:
        # 创建转换器并执行转换
        converter = Bag2LeRobot(
            bag_path=args.bag_path,
            output_root=args.output_dir,
            task_language=args.language,
            sync_topic=args.sync_topic,
        )

        converter.convert()

    except Exception as e:
        logging.error(f"转换失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断运行")
        sys.exit(0)
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        sys.exit(1)
