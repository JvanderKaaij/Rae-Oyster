import sys
if sys.prefix == 'C:\\Users\\Joey\\miniforge3\\envs\\ros_env':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = 'D:\\UserProjects\\Joey\\Robotics\\Rae\\rae-ros-oyster\\rae_oyster\\install\\rae_oyster'
