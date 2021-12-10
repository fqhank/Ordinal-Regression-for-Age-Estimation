# Ordinal-Regression-for-Age-Estimation
### CVPR 2016 Ordinal Regression with Multiple Output CNN for Age Estimation

原文链接：[ordinal regression](https://openaccess.thecvf.com/content_cvpr_2016/html/Niu_Ordinal_Regression_With_CVPR_2016_paper.htm)

该论文提出利用顺序回归的方法，结合多输出CNN网络，实现人脸的年龄估计任务。
作为练手项目进行复现练习。从网络结构、数据集调用、train、validation以及test等都亲自完成。代码基于pytorch实现（原论文采用caffe）

### some points to notice

1.模型采用了多头输出，需要构造多达**56**个**[80,2]**全连接层。直接在`__init__`中逐个定义显然是不现实的，可以采用exec函数循环执行定义不同命名的全连接层，也可以首先创建一个空list，循环append创建的nn.linear实例，后续只需按index调用即可（推荐）；

2.`pytorch.nn`的cross entropy loss实际上首先执行了softmax，再取了log对数，最后计算交叉熵，故网络输出的logits不需要再单独计算softmax。**但是**！强烈建议自己编写不带softmax的loss函数，而在网络输出处自行添加，方便复现文中设计的task importance模块，同时避免在计算平均误差时出现逻辑混乱的情况；

3.论文提出的网络相当简单，所以调参和一些关键设置成为了结果的关键（对结果影响非常大）

实验证明：

​	（1）SGD的泛化能力强于Adam，且weight_decay在SGD上的效果较在Adam上更明显。Adam会使得网络收敛的更快，但SGD最终取得了更好的结果。建议采用SGD；

​	（2）Data Augmentation对结果影响非常大。在没有使用前（仅使用`ToTensor()`以及移动至[-1,1]），最佳的结果是出现在weight_decay=0.01下的3.70，此时较大的正则化使得网络难以继续学习，而更小的weight_decay则会明显使得网络出现过拟合，泛化能力不增反降。在添加了一些常规的augmentation之后，weight_decay=0.0001时能够取得3.44的结果，说明此时L2正则化对网络的限制刚刚好，而数据增强又提高了泛化能力。（从结果看网络仍有进一步提升空间，有待探索）；

​	（3）关于学习率的设置。学习率的降低是十分必要的，在学习效果停止增长时降低学习率可以明显带来进一步提升。但是合适的降低策略需要探索，在尝试了指数、阶梯和自动检测等多种策略后选定了阶梯式，且降低率也需要一步步摸索；

​	（4）全连接层后的dropout并不必要，似乎对结果还有一定损伤；

​	（5）batch size不一定越大越好。虽然有的教程指出在GPU能力范围内越大的size可以帮助越快的收敛，但采用1024的batch使得网络在val和test上的表现不如预期**（有待进一步验证）**；

​	（6）在没有采用Data Augmentation时发现使用early stop and go back在一定程度上可以阻止网络过拟合，显著提升泛化效果。但使用数据增强后发现，这一策略严重损伤了最终结果。分析认为在数据增强后，网络的学习效果实际上在持续提升，stop and go back会阻碍这一过程。



