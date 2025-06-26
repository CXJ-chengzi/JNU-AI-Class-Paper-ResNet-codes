# 基于CIFAR-10的ResNet-18算法复现
### `Pytorch`实现：使用`ResNet-18`网络训练`Cifar10`数据集，测试集准确率达到95.23%(使用预训练模型)

本文是用ResNet-18网络训练CIFAR-10数据集的，下面是对此项目的具体说明。
#### 1 代码环境
python == 3.9.23 <br/>
torch == 1.10.2 <br/>
torchvision == 0.11.3 <br/>
numpy == 1.23.0 <br/>
Matplotlib == 3.5.1<br/>
Scikit-learn == 1.0.2<br/>
TQDM == 4.64.0<br/>

#### 2 项目结构
这里对整个的项目结构进行说明：<br/>
1.test.py和train.py分别为测试代码和训练代码；<br/>
2.visualization.py是数据集可视化文件；<br/>
3.performance_indicators.py是性能指标文件，会对算法的一些性能进行可视化，同时会输出具体数据；<br/>
4.flowchart.py文件运行可以得到ResNet-18的算法流程图；<br/>
5.evaluation.py是算法在不同场景下的测试代码，并进行了可视化；<br/>
6.在utils文件里面存放了三个文件：<br/>
cutout.py用于cutout数据增强；readData.py用于读取数据；ResNet.py是ResNet-18的代码。<br/>

#### 3 运行指南
若需运行本项目代码，首先需要安装好软件环境，python版本和各种学习库的版本要安装一致，当然如果兼容也是可以的。<br/>
在环境配置好的情况下，要下载CIFAR-10的dataset，然后放在项目目录里面，不主动下载的话，在代码运行的时候也会自动下载的。<br/>
在运行所有代码之前，要在项目目录下创建cheakpoint文件夹，这是存放训练模型的地方。<br/>
之后可以正常运行代码：先readData获取数据，当然也可以不这么做，后面训练时会直接调用（还是会运行）。然后通过train.py在训练集上面训练模型，训练完成之后通过test.py在测试集上面测试模型。<br/>
注意：下载代码的时候要要注意data路径,这里的要和你存放dataset的路径是一致的，当然如果你是代码自动下载的，那么会直接存放在项目文件中，无需修改。<br/>
```python
pic_path='dataset'
```
<br/>
其他的文件visualization.p、evaluation.py、flowchart.py、performance_indicators.py可在数据加载完成和模型训练完成之后自行运行，都是可以直接运行的。<br/>
如果训练时间过长，推荐选用好的电脑（因为有好的GPU可以大幅提升训练速度，笔者的电脑不太行，跑一次6h），或者一开始训练轮次epoch小一点，推荐100。如果仍然觉得时间过长，可以使用预训练模型，该模型是由Github作者ZOMIN提供的：<br/>
百度云盘：https://pan.baidu.com/s/1yKXWWf1UEXS_gsWnM6sFDA 提取码：z66g。<br/>

#### 4 参考资料
本人在学习人工智能相关知识，复现ResNet-18算法时，查阅了很多资料，包括但不限于以下列出的这些，在这里向这些论文和网络资料的作者表示特别的感谢，有你们我才能顺利完成我的课程论文。<br/>
[1].人民网.《抢抓人工智能发展的历史性机遇》，2025：http://opinion.people.com.cn/n1/2025/0224/c1003-40423933.html.<br/>
[2].知乎.《人工智能的重要性与影响》，2024：https://zhuanlan.zhihu.com/p/12821795210.<br/>
[3].A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.<br/>
[4].Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural computation, 1989.<br/>
[5].He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016.<br/>
[6].He K, Zhang X, Ren S, et al. Identity Mappings in Deep Residual Networks[J]. arXiv preprint arXiv:1603.05027, 2016.<br/>
[7].He K, Zhang X, Ren S, et al. Residual Networks Behave Like Ensembles of Relatively Shallow Networks[J]. arXiv preprint arXiv:1605.06431, 2016.<br/>
[8].知乎.《Resnet到底在解决一个什么问题呢？》2019:https://www.zhihu.com/collection/742455948.<br/>
[9].CSDN：https://blog.csdn.net/qq_41185868/article/details/82793025.<br/>
[10].https://zhuanlan.zhihu.com/p/515734064.<br/>
[11].https://blog.csdn.net/m0_64799972/article/details/132753608.<br/>
