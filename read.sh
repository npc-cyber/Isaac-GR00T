# 激活环境
conda activate gr00t

# 安装项目依赖（包括 gr00t 模块）
pip install -e .

# ros2 消息编译 
source bullet_data/install/setup.sh     

# 不需要 pip install -e .
export PYTHONPATH="$PWD:$PYTHONPATH"
# 关闭的时候 先关闭 gui
