# RM-Standard-Robot
The code for my RoboMaster Standard Robot, including STM32\OpenCV\TensorFlow.

文件详情：
1. 底盘v1.3.2是用于RM开发板A的程序，用st-link烧入。若需要用CubeMX重建工程，请阅读文件夹内的README文件，否则会导致车辆失控。
2. input_data.py是用于下载MNIST数据集文件，请更改下载路径以应用。
3. deep2.py是训练MNIST数据集的程序。
4. deep4.py是识别all_rev.png图片的程序。
