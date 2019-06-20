# RM-Standard-Robot
===
The code for my RoboMaster Standard Robot, including STM32\OpenCV\TensorFlow.
---
文件详情：
---
#1. 底盘v1.3.2是用于RM开发板A的程序，用st-link烧入。若需要用CubeMX重建工程，请阅读文件夹内的README文件，否则会导致车辆失控
#2. input_data.py是用于下载MNIST数据集文件，请更改下载路径以应用
#3. deep2.py是训练MNIST数据集的程序
#4. deep4.py是识别all_rev.png图片的程序
#5. 激光笔数字指向文件夹  
##5.1 ok.jpg是测试的演示图片  
##5.2 deep5.py是外接摄像头识别程序，该程序有单目测距功能  
##5.3 deep6.py是在deep5的基础上增加了通过串口发送舵机控制数据的程序  
##5.4 stm32f103是解析舵机控制数据并控制舵机和激光笔指向数字的程序  
##5.5 servo_test.py可用于测试电脑与stm32通讯并控制舵机  
