# ArmDataCollect.msg

std_msgs/Header header  # 头部信息，包括时间戳和坐标系

string arm_name  # "left" 或 "right" 

# 运动位置 - 笛卡尔空间
geometry_msgs/PoseStamped pose  # 目标位置和姿态（笛卡尔空间）

string rotation_type  # 旋转类型： 'axis_angle', 'euler_angles', 'quaternion', 'rotation_6d', 'matrix'

# 关节角 - 关节空间
string[] joint_name  # ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
float64[] joints_angle  # 每个关节的角度（数组）
# float64[] velocity  # 每个关节的速度（数组）
# float64[] torque  # 每个关节的扭矩（数组）

float64 gripper_distance  # 夹爪开合度 0-1之间值ros

# sensor_msgs/Image left_image
sensor_msgs/Image right_image 
sensor_msgs/Image top_image