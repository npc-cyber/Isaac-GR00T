# SynchronizedData.msg

std_msgs/Header header  # 时间戳和坐标系信息

ArmDataCollect left_arm_data  # 左臂数据
ArmDataCollect right_arm_data  # 右臂数据

sensor_msgs/Image left_image  # 左图像数据
sensor_msgs/Image right_image  # 右图像数据
sensor_msgs/Image top_image  # 顶部图像数据
