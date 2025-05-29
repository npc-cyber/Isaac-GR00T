import rosbag2_py
from rclpy.serialization import deserialize_message
from common_msgs.msg import ArmDataCollect
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image


def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


def main():
    bag_path = '/mnt/zhihui/code/embed_datasets/rosbag2_2025_04_24-07_49_25/rosbag2_2025_04_24-07_54_28/rosbag2_2025_04_24-07_54_28_0.db3'  # 替换为你的 ros2 bag 包的实际路径
    storage_options, converter_options = get_rosbag_options(bag_path)

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == '/right_arm_data_cmd':
            msg_type = type_map[topic]
            if msg_type == 'common_msgs/msg/ArmDataCollect':
                msg = deserialize_message(data, ArmDataCollect)
                # 解析头部信息
                header: Header = msg.header
                print(f"Header: stamp={header.stamp}, frame_id={header.frame_id}")

                # 解析机械臂名称
                arm_name = msg.arm_name
                print(f"Arm name: {arm_name}")

                # 解析笛卡尔空间位置和姿态
                pose: PoseStamped = msg.pose
                position = pose.pose.position
                orientation = pose.pose.orientation
                print(f"Pose: position=(x={position.x}, y={position.y}, z={position.z})")
                print(f"Orientation: (x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w})")

                # 解析旋转类型
                rotation_type = msg.rotation_type
                print(f"Rotation type: {rotation_type}")

                # 解析关节角信息
                joint_names = msg.joint_name
                joint_angles = msg.joints_angle
                for i in range(len(joint_names)):
                    print(f"Joint {joint_names[i]} angle: {joint_angles[i]}")

                # 解析夹爪开合度
                gripper_distance = msg.gripper_distance
                print(f"Gripper distance: {gripper_distance}")

                # 解析图像信息
                # 这里只是简单打印图像的高度和宽度，你可以根据需求进一步处理图像数据
                right_image: Image = msg.right_image
                top_image: Image = msg.top_image
                print(f"Right image: height={right_image.height}, width={right_image.width}")
                print(f"Top image: height={top_image.height}, width={top_image.width}")


if __name__ == '__main__':
    main()