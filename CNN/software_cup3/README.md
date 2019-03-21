### 第一部分
1.训练集边缘检测<br/>
2.通过边缘检测结果进行数字切割(这部分的算法使用OpenCV进行。使用不同的算子进行边缘检测，选择最佳的算子。)<br/>
3.进行训练（参照MINIST训练方法）<br/>
4.获取数字识别训练结果<br/>
### 第二部分
1.信用卡文字定位：{使用CNN-object detection 进行文字区域定位}<br/>
    1.1训练数据集标记：使用labelme进行数据标记。具体使用方法参考github。<br/>
    1.2训练：输入图像：960*720， 输出：y=(p, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y)^T
    1.3损失函数定义：if y0 == 1: loss_function = sigma(i=0:9)(y_i_ - y_i)^2<br/>
        if y0 == 0: loss_function = (y_i_ - y_i)^2
2.信用卡卡号切割，切割方法和上面一样<br/>
3.切割后的图片送入上面训练的模型中进行检测<br/>