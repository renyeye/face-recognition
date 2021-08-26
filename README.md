# face-recognition
此项目为人工智能训练营课程作业

以其他作者原项目为基础，进行一些改进和拓展（原项目：https://github.com/fengpanhe/AI-Experiment）

# 内容说明
1.需要用到以下python模块

pillow  numpy   matplotlib  tensorflow

2.数据集来源

faces_4为灰白数据集，源自：http://www.cs.cmu.edu/~tom/mlbook.html

faces_colorful为彩色数据集，源自CelebA，作者从其中筛选了男女各40张图片

3.内容

原项目face_recognition进行的是灰白人脸图片的识别

face_recognition优化显示、face_recognition更换模型、face_recognition识别彩图  为小组对原项目进行的改进和拓展，
其中：

优化显示：主要针对程序可视化输出进行优化，在原折线图的基础上，增加显示每张图的预测结果

更换模型：利用训练营day2讲过的深度学习神经网络模型，重复实现原功能

彩图识别：对模型稍作调整，以识别彩色人脸图

# 注意事项
1.路径修改

运行不同项目需要修改config.py

运行原项目、优化显示和更换模型为：
  'IMGPATH': '../faces_4/'  

运行彩图识别为：
'IMGPATH': '../faces_colorful/'

2.彩色人脸识别

彩图识别项目的运行结果尚不佳，模型需要改进
