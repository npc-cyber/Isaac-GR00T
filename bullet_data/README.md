# teleoperation_data_convert

## 批量转换rosbag包到N1（lerobot + modality.json）格式数据
```bash
python bag2lerobot_v2.py --bag_path ./embed_datasets/rosbag2_2025_04_24-07_49_25 --output_dir ./ur3e 
python bag2lerobot_v2.py --bag_path ../../../rosbag2_2025_05_26-13_49_19/ --output_dir ../../../panda 
```
### 输入bag包目录结构
```
.
├── rosbag2_2025_04_24-07_54_28
│   ├── metadata.yaml
│   └── rosbag2_2025_04_24-07_54_28_0.db3
├── rosbag2_2025_04_24-07_55_08
│   ├── metadata.yaml
│   └── rosbag2_2025_04_24-07_55_08_0.db3
```
### 输出N1格式数据
```
├── data
│   └── chunk-000
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       ├── episode_000002.parquet
│       ├── episode_000003.parquet
│       ├── episode_000004.parquet
│       ├── episode_000005.parquet
│       ├── episode_000006.parquet
│       ├── episode_000007.parquet
│       └── episode_000008.parquet
├── meta
│   ├── episodes.jsonl
│   ├── info.json
│   ├── modality.json
│   └── tasks.jsonl
└── videos
    └── chunk-000
        ├── observation.images.head_view
        └── observation.images.right_view
```
## 检查bag包数据
```bash
python get_rosbag.py 
```
### bag包内容示例
```
Header: stamp=builtin_interfaces.msg.Time(sec=1745481289, nanosec=424264960), frame_id=
Arm name: right_arm
Pose: position=(x=-0.22122700783848048, y=-0.40873340420181403, z=0.2706924926309573)
Orientation: (x=-0.005463278857700826, y=-0.3856399888114096, z=-0.05662929695032316, w=0.9208936281355035)
Rotation type: 
Joint right_shoulder_pan_joint angle: 0.9076099395751953
Joint right_shoulder_lift_joint angle: 0.3426670271107177
Joint right_elbow_joint angle: -1.3258042335510254
Joint right_wrist_1_joint angle: -0.07624860227618413
Joint right_wrist_2_joint angle: 2.2069859504699707
Joint right_wrist_3_joint angle: 0.7088466286659241
Gripper distance: 0.1450980392156863
Right image: height=480, width=848
Top image: height=360, width=640
```