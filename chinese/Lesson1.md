# 第一课
[主页](http://course-v3.fast.ai/) / [视频](https://youtu.be/BWWm4AzsdLk) /  [论坛第一课板块](https://forums.fast.ai/t/lesson-1-official-resources-and-updates/27936) / [论坛](https://forums.fast.ai/t/faq-resources-and-official-course-updates/27934/1)



## 欢迎! 

确保已经设置好了GPU环境并且可以运行Jupyter Notebook

[00_notebook_tutorial.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/00_notebook_tutorial.ipynb)


快捷键:

- <kbd>Shift</kbd>+<kbd>Enter</kbd>: 运行单元里的代码，选中下个单元

- <kbd>Up Arrow</kbd>+<kbd>Down Arrow</kbd>: 在单元间切换

- <kbd>b</kbd>: 创建新单元

- <kbd>0</kbd>+<kbd>0</kbd>: 重启内核



[[2:45](https://youtu.be/BWWm4AzsdLk?t=165)] 

Jupyter Notebook 对于数据科学工作者来说是一个有趣的工具，你不仅可以获得静态的信息，还可以进行交互实验.

基于过去三年的经验，这样使用notebook和学习材料最有效:

直接从头到尾看视频. 
   - 不必试图跟上课程内容，课程节奏的被设计地比较快，这样你可以快速地对各方面有一个初步的概念，并且了解到这些方面是怎样结合在一起的
   - 看完第一遍后，再回过头来，重新看一遍视频，在需要的地方暂停，按照视频里讲的自己做一遍。确保你可以做出和我一样的结果，然后在这个基础上用你自己的方式做些拓展.
   - 第一遍看视频时不要运行代码，中间不要停下来，不必试图理解所有内容. 



### 你可以做世界顶尖水平的深度学习 [[4:31](https://youtu.be/BWWm4AzsdLk?t=271)]

![](/lesson1/1.png)

在这里获取课程资源:
- [http://course-v3.fast.ai/](http://course-v3.fast.ai/)
- [https://forums.fast.ai/](https://forums.fast.ai/latest)





### 为什么听Jeremy讲课 [[5:27](https://youtu.be/BWWm4AzsdLk?t=327)]

![](/lesson1/2.png)





### 使用机器学习做有用的事情 [[6:48](https://youtu.be/BWWm4AzsdLk?t=408)]

![](/lesson1/3.png)



[[7:26](https://youtu.be/BWWm4AzsdLk?t=446)]

![](/lesson1/4.png)

如果你能坚持7周，每周花10个小时在课程上，最终你可以做到：

1. 为你选择的图片构建一个世界水平的分类器
2. 对任何你感兴趣的数据集做文本分类
3. 做商业预测，比如销售预测
4. 构建像Netflix一样的推荐系统

所有这些并非玩具示例，而是可以在Kaggle竞赛赢得前十的实战项目，这样的成绩足以超过学术社区. 

前提条件是一年的编程经验和高中数学知识.






### 关于深度学习的一些无意义或者不正确的说法 [[9:05](https://youtu.be/BWWm4AzsdLk?t=545)]

![](/lesson1/5.png)

- 它不是一个黑盒子，它的工作是可以解释的.
- 大部分实际的应用不需要太多数据.
- 想掌握深度学习，并不需要你有一个博士学位，当然如果你有的话，也不会妨碍你从事深度学习.
- 可以被广泛应用在很多不同的领域，并不单单是视觉领域
- 你不需要太多硬件，36美分一小时的服务器足够求解大部分问题，并取得世界级的结果.
- 它确实不能帮助你构建一个有意识的大脑，这并不是我们关心的，我们专注于解决有趣的现实问题.



[[10:24](https://youtu.be/BWWm4AzsdLk?t=624)]

![](/lesson1/6.png)

棒球 VS. 板球 - Nikhil 提供的一个例子，学完第一课后，你将能够完成这个题目



### 由上至下的学习方式 [[11:02](https://youtu.be/BWWm4AzsdLk?t=662)]

![](/lesson1/7.png)

我们从阅读代码开始，这不同于大部分学校课程。今天我们将学习构建有用的程序。今天的课程结束后，你并不会学会所有的理论，有很多东西你可能并不清楚它们是怎样工作的，以及它们为什么是有效的。这没有关系，在接下来的7周里，你将会学会这些。现在，着手写代码而不是关注理论知识，这是更有效的学习方式



## 什么是你的宠物 [[12:26](https://youtu.be/BWWm4AzsdLk?t=746)]

[lesson1-pets.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb)

<kbd>Shift</kbd>+<kbd>Enter</kbd> 运行一个单元

每一个notebook都以这三行代码开始:
```
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```
`%` 开头的代码是对Jupyter Notebook的特殊指令, 不是python代码. 它们被称作"魔法"

- 如果程序执行时有人修改了依赖库的代码, 自动重新加载库
- 如果有人要求画出一些内容，就把它们画在Jupyter Notebook上

接下来的两行引入了fastai库:

```python
from fastai import *
from fastai.vision import *
```

什么是fastai库? [http://docs.fast.ai/](http://docs.fast.ai/)

我们的所有工作都是基于fastai库或者[PyTorch](https://pytorch.org/)，fastai库本身也是基于PyTorch的。 PyTorch 是一个很流行的库，用户在快速增长。几年前我们使用TensorFlow，但是我们发现PyTorch用起来更快捷，所以我们改用了PyTorch。 

目前fastai库支持四种应用:

1. 计算机视觉
2. 自然语言文本
3. 表格数据
4. 协同过滤


[[15:45](https://youtu.be/BWWm4AzsdLk?t=945)]

`import *` - 你们被告知永远不要这样做.

对于很多库来说，不在生产环境使用`import *` 是有充分的理由的。但对MATLAB这样的库来说，是另外一回事。所有的东西都已经准备好了。你不必多次引入各种库。这是一个有趣的现象，我们走了两个极端。在科学计算界里有一种方式，在软件工程界有另外一种方式。这两种方式都有充足的理由。 

对于fastai库，我们支持两种方式。当你想用Jupyter Notebook快速做些尝试时， 你不希望总是回到文件开头添加一些引用。你希望有很多代码能自动完成，做很多实验性的工作, 所以 `import *` 是合适的。当你构建生产环境程序时，你可以遵守PEP8这样的软件工程规范。这是一种不同的编程风格。数据科学编程并非没有规范，只不过是这里的规则不同于软件工程。在训练模型时，最重要的事情是能够快速尝试，所以你将看到很多不同的流程，风格和工具 。这是有原因的，慢慢地，你将了解到这些原因。

另外，fastai库是用非常有趣的模块化方法设计的，使用import *, 带来的问题会比你预期的少，它被特意设计，使得你能够非常方便地加载和使用里面的方法。

数据 [[17:56](https://youtu.be/BWWm4AzsdLk?t=1076)]

课程中，我们主要从这两个地方获取数据:

1. 学术数据集
    - 学术数据集是很重要的。它们很有趣。学者们花费了大量时间搜集整理一个数据集，用来测试各种不同的算法在数据集上的表现。他们设计一个数据集，然后用不同的方式挑战，寻求能够获得突破，得到更好的结果。 
    - 我们将从一个名为宠物数据的学术数据集开始
2. Kaggle竞赛数据集
  

这两种数据集都是我们关注的，它们提供了有力的基线，可以很好地评价你做得如何。使用来自竞赛的Kaggle数据集，你可以把结果提交到Kaggle，看下你在竞赛中的成绩。如果你能进入前10%，那说明你做得相当好。

对于学术数据集，学者们在论文里写下了他们在这个数据集上取得的最好结果。这就是我们将要做的。我们创建争取能够在Kaggle竞赛里取得高排名的模型，最好是前十，而不仅是前10%，或者能达到或超过学术论文里发表的最佳结果。在你使用一个学术数据库时，把它放在参考文献里。你不必立即阅读这论文，当你希望了解到数据集的更多信息，它为什么被创建，是如何被创建的，这时你可以从中了解到细节。 

宠物数据集要求我们区分出37种不同的猫和狗的品种。这是一个困难的任务。事实上，在之前的课程里，我们使用另外一个数据集，它只要求分辨出图片里是狗还是猫。所以你有一半的机会猜对，并且猫和狗的区别很大。而不同品种的猫狗之间的差别并不大。 

为什么我们更换了数据集？我们意识到，深度学习很快很简单，现在区分猫和狗这个问题对深度学习来说太简单了，尽管几年前这被认为是非常困难的，达到80%的准确率就是最好水平。我们的模型可以在不做任何优化的情况下直接输出完全正确的结果，这样我们就不能讲授一些复杂的技术，所以今年我们选择了一个更难的问题。 



[[20:51](https://youtu.be/BWWm4AzsdLk?t=1251)]

学术上，区分相似分类被叫做细粒度分类.  

### untar_data

我们要做的第一件事情是下载和解压数据，我们将使用`untar_data`这个函数，它会自动下载并解压数据。AWS给我们提供了许多空间和带宽，数据集可以被很快的下载下来。

```python
path = untar_data(URLs.PETS); path
```

### help 

怎样可以知道`untar_data`是什么？你可以输入help，这样你可以看到这个方法来源于那个模块(因为我们使用了`import *`，所以你不必知道所属的模块), 它会做什么, 以及一些你以前不知道的事情，或许你是一个有经验的开发者，但可能并不清楚究竟应该传递什么参数。你可能很熟悉这些名字: url, fname, dest, 但你可能没怎么见过`Union[pathlib.Path, str]`. 这是参数的类型，如果你熟悉类型编程语言的话，你可能会经常见到它们,但Python开发者可能对它不怎么熟悉。只有你知道了每一个传入的参数的类型，你才能知道怎样使用一个函数，所以我们在帮助里说明了类型信息。

对于这个函数, `url` 是一个字符串, `fname` 是一个 path 或者一个字符串，默认是None(`Union` 表示 "或者"). `dest` 是一个 path 或者一个字符串

```python
help(untar_data)
```

```
Help on function untar_data in module fastai.datasets:

untar_data(url:str, fname:Union[pathlib.Path, str]=None, dest:Union[pathlib.Path, str]=None)
    Download `url` if doesn't exist to `fname` and un-tgz to folder `dest`
```

从代码里可以看到，我们不需要传递文件名 `fname` 和目录 `dest`参数，程序会根据url自动生成这些参数。稍后我们会学习如何获取关于这个功能的详细文档

对于课程里的所有数据集，我们都定义了对应的常量。在这个 [URLs](https://github.com/fastai/fastai/blob/master/fastai/datasets.py) 类里，你可以看到程序是怎样下载数据的。

`untar_data` 会下载数据集到一个方便使用的目录，把数据集解压，返回文件路径。 

```python
path = untar_data(URLs.PETS); path
```
```
PosixPath('/data1/jhoward/git/course-v3/nbs/dl1/data/oxford-iiit-pet')
```
在Jupyter Notebook里，你可以仅仅写一个变量 (分号只是一个python语句的结尾)，它就可以被打印出来。你也可以用`print(path)`，但我们尽可能用最简便的方式，所以只写了一个变量。这里输出的就是数据的路径。 

下次再执行这段代码时，因为数据以及被下载过了，它就不会被重复下载。它以及被解压了，也就不会被重复解压。所有的功能都被设计得很简便，很自动。

[[23:50](https://youtu.be/BWWm4AzsdLk?t=1430)]

在Python里，有些语法不是很方便交互。比如，对一个路径对象，想查看路径下的文件，需要不少代码。所以，我们为python对象拓展了一些方法。其中一个就是为path添加了一个`ls()`方法

```python
path.ls()
```
```
['annotations', 'images']
```

这是目录下的文件。我们刚刚下载下来的东西。 

### Python 3 pathlib [[24:25](https://youtu.be/BWWm4AzsdLk?t=1465)]

```python
path_anno = path/'annotations'
path_img = path/'images'
```

如果你不是一个经验丰富的Python开发者，你可能会不太熟悉斜杠的这种用法。这是Python 3里一个非常方便的函数。它是在[pathlib](https://docs.python.org/3/library/pathlib.html)定义的。Path对象比字符串好用得多。使用它你可以这样创建子目录。无论你用的是 Windows, Linux, 还是 Mac，它的行为都是一样的。`path_img` 数据集里图片的路径。

[[24:57](https://youtu.be/BWWm4AzsdLk?t=1497)]

如果你想用一个新的数据集做深度学习，第一件事情大概就是看看数据集里有什么。我们可以看到数据集里有`annotations` 和 `images` 两个目录，来看下images 目录里有什么? 

### get_image_files [[25:15](https://youtu.be/BWWm4AzsdLk?t=1515)]

get_image_files 可以取到一个包含所有图片路径的数组. 

```python
fnames = get_image_files(path_img)
fnames[:5]
```
```
[PosixPath('/data1/jhoward/git/course-v3/nbs/dl1/data/oxford-iiit-pet/images/american_bulldog_146.jpg'),
 PosixPath('/data1/jhoward/git/course-v3/nbs/dl1/data/oxford-iiit-pet/images/german_shorthaired_137.jpg'),
 PosixPath('/data1/jhoward/git/course-v3/nbs/dl1/data/oxford-iiit-pet/images/japanese_chin_139.jpg'),
 PosixPath('/data1/jhoward/git/course-v3/nbs/dl1/data/oxford-iiit-pet/images/great_pyrenees_121.jpg'),
 PosixPath('/data1/jhoward/git/course-v3/nbs/dl1/data/oxford-iiit-pet/images/Bombay_151.jpg')]
```

 把所有的文件放在一个文件夹下，这是处理计算机视觉数据集的常见方式。下一步有趣的事是获取标签。在机器学习中，标签是指我们想要预测的东西。如果我们浏览下这些文件，可以看出标签就是文件名的一部分。 文件名的格式是 `目录/标签_编号.拓展名`。我们要想办法获取到文件名中`标签`部分的列表, 这样就可以得到标签。这些就是用来构建一个深度学习模型的所有东西:
 - 图片文件
 - 标签

在fastai里，这被设计的很简单。有一个叫做`ImageDataBunch`的对象，一个ImageDataBunch 代表你创建一个模型所需要的所有数据，使用工厂方法可以很方便得创建一个包含训练集和验证集的ImageDataBunch，训练集和验证集里都包含图片和标签。 

在这个案例里，我们需要从文件名提取标签。我们会使用 `from_name_re`. `re` 是python里做正则表达式的模块，正则表达对提取文本是非常有用的。这是提取标签的正则表达式:

```python
np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'
```
对于这个工厂方法，我们可以传入这些参数

- path_img: 存放图片的目录
- fnames: 存放文件名的列表
- pat: 从文件名中提取标签的正则表达式
- ds_tfm: 变形，我们稍后再讲
- size: 你想处理的图片的尺寸
  

图片需要按固定的尺寸来处理，这看起来有点怪。这是目前深度学习技术的一个缺点。为了处理地更快，GPU需要对同时处理的一组数据执行相同的指令。如果图片的形状尺寸不同，这就做不到了。所以我们需要让所有图片有相同的形状尺寸。在课程的第一部分，我们会把图片做成正方形。第二部分，我们会学习使用矩形。这两者有些奇特的区别。很多的人在很多计算机视觉模型里都是用的正方形的方法。224 * 224，是一个极其常见的尺寸，大部分模型都使用这个尺寸，如果你直接使用这个尺寸，大部分情况下很容易得到不错的结果。原因我们晚些再介绍。这是我想教给大家的一个小技巧，它通常很有效。你们尽管使用这个尺寸，在大部分情况下它都是有效的。



```python
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224)
data.normalize(imagenet_stats)
```

[[29:16](https://youtu.be/BWWm4AzsdLk?t=1756)]

`ImageDataBunch.from_name_re` 将会返回一个dataBunch对象。在fastai里，所有你用来建模的东西都是一个DataBunch对象。DataBunch对象基本上会包含 2个或者3个数据集 - 训练数据，验证数据，有时会有测试数据。每一个都包含图片和标签，或者文本和标签，或者表格和标签，等等。它们都被放在一个地方(比如 `data`)。

一个需要多说明一点的是标准化。在几乎所有机器学习任务里，你需要让数据有相同的“尺寸”，就是说有相同的平均值和标准差。所以fastai里有一个标准化函数，我们可以用这样的方式来标准化数据。



[[30:25](https://youtu.be/BWWm4AzsdLk?t=1825)]

提问：如果图片尺寸不是224，这个函数会做些什么？ 

我们稍后会学习这部分内容。变形操作会对图片做一些处理，其中一个就是把图片尺寸编程224。 


### data.show_batch
我们来看看这些图片。 这是data bunch里的一些图片。data.show_batch可以显示data bunch里的内容。可以看出，这些图片都被恰当地缩放或者裁剪过。这种方法叫中心裁剪，它抓取出图片的中间部分，然后调整图片尺寸。我们将会学习这个方法的更多细节，它是相当重要的。 它基本上是裁剪和调整尺寸两种方法的组合。

```python
data.show_batch(rows=3, figsize=(7,6))
```
![](/lesson1/8.png)

我们也会用它来做数据增强。对于裁剪多少和在哪里裁剪之类的问题，是有些随机性的。 

基本思路是裁剪，缩放和设置边距。做数据增强时，根据不同情况有很多不同的方式，我们稍后将会学习这些内容。 


[[31:51](https://youtu.be/BWWm4AzsdLk?t=1911)]

**提问**: 标准化图片是什么意思？ 

后面我们会学习更多关于标准化图片的知识。简单来讲，一个像素有红绿蓝三个通道，每个值介于0到255之间，有的通道会太亮，有的会太暗，有的变化比较大，有的没什么变化。如果我们让三个通道的均值都是0，标准差都是1，会有利于我们训练深度学习模型。 

如果数据没有标准化，难以训练出好的模型。如果你的模型不效果不好，需要检查下是否标准化了数据。


[[33:00](https://youtu.be/BWWm4AzsdLk?t=1980)]
**提问**: GPU内存是2的指数，相比于224来说，是不是256对于利用GPU更有效？

简单来说，模型的最后一层的尺寸是7*7，所以我们希望的输入是7乘以2的指数。


[[33:27](https://youtu.be/BWWm4AzsdLk?t=2007)]

我们将学习所有这些细节。但重要的是，我希望能尽快开始训练模型。 

### 查看数据是重要的

一个优秀的从业者的一个重要能力就是能够查看数据。使用`data.show_batch`方法来看看数据是很重要的。 当你查看你拿到的数据时，你会发现这些情况会很常见：图片有奇怪的黑色边框，在图片中的一些物品上有文字，有些图片被旋转过。一定要查看下这些数据。

另外一件我们要做的是查看标签。所有可能的标签都是一种分类，使用DataBunch，你可以打印`data.classes`

```python
print(data.classes)
len(data.classes),data.c
```

```
['american_bulldog', 'german_shorthaired', 'japanese_chin', 'great_pyrenees', 'Bombay', 'Bengal', 'keeshond', 'shiba_inu', 'Sphynx', 'boxer', 'english_cocker_spaniel', 'american_pit_bull_terrier', 'Birman', 'basset_hound', 'British_Shorthair', 'leonberger', 'Abyssinian', 'wheaten_terrier', 'scottish_terrier', 'Maine_Coon', 'saint_bernard', 'newfoundland', 'yorkshire_terrier', 'Persian', 'havanese', 'pug', 'miniature_pinscher', 'Russian_Blue', 'staffordshire_bull_terrier', 'beagle', 'Siamese', 'samoyed', 'chihuahua', 'Egyptian_Mau', 'Ragdoll', 'pomeranian', 'english_setter']

(37, 37)
```

这是我们使用正则表达式找出的所有的可能的标签。之前我们讲过有37中可能的分类，检查下`len(data.classes)`，确实是 37。DataBunch也有一个叫做 `c`的属性。我们稍后再学习这些技术细节，现在你可以把它理解为类别的数量。对于回归和多标签分类问题，这不是很准确，但对于目前这样的解释够用了。 `data.c` 是一个很重要的信息，这点要记住，它基本上表示类别的数量，至少对于分类问题是这样的。 

 ## 训练 [[35:07](https://youtu.be/BWWm4AzsdLk?t=2107)]

信不信由你，我们现在已经准备好训练模型了。在fastai里我们使用 "learner" 来训练模型。

 - **DataBunch**: 一个fastai里广泛使用的概念，代表你的数据。对具体的应用，有对应的子类，比如ImageDataBunch
 - **Learner**: 一个fastai里广泛使用的概念，代表学习拟合一个模型的操作。 在各种具体的应用中有很多对应的子类，用来简化使用，比如有一个convnet learner，可以用来创建一个卷积神经网络。

```python
learn = create_cnn(data, models.resnet34, metrics=error_rate)
```

目前，创建一个卷积神经网络的learner，只需要知道两个参数:
`data`: 你的数据，一个data bunch.
`arch`: 模型的结构。有很多不同的方式构建一个卷积神经网络. 

现在，你需要知道有一种叫做ResNet的模型，它几乎总是非常有效的。你只需要选择ResNet的大小，ResNet有ResNet34 和 ResNet50两种大小。当我们开始做一个任务时，我会先用这个小些的模型，它训练地更快。想成为一个好的实践者，现在只需要知道有两种架构效果很好：ResNet34 和 ResNet50。先试下小的，看看效果是否足够好。

这是我们要创建一个卷积神经网络learner所需要知道的所有信息。 

另外一个传入的参数是metrics（度量），metrics是训练时逐个地打印出来的东西。传入error_rate就是让它打印出错误率。

[[37:25](https://youtu.be/BWWm4AzsdLk?t=2245)]

![](/lesson1/c1.png)

第一次，我在一个新安装的环境里运行这些代码，它下载了 ResNet34 预训练权重。也就是说，这是一个针对真实任务训练过的实战模型。这个任务是训练模型看50万张各种物品的图片，这些物品属于1000个种类，这个图片数据集叫ImageNet。所以我们可以下载这些预训练过的参数，不必从一个一无所知的模型，而是从一个已经能够识别ImageNet里1000种类别物品的模型开始。并非所有37种品种都在ImageNet里，但里面确实有几种猫和几种狗。所以这个模型知道一些猫狗的品种，并且知道很多动物和很多照片。所以我们不需要从一个空模型开始，而是基于一个已经懂得识别一些图片的模型。预训练模型会在第一次被使用时被自动下载，以后就不会再下载了，而是直接使用先前下载的那个。 

## 迁移学习 [[38:54](https://youtu.be/BWWm4AzsdLk?t=2334)]

这部分很重要。我们将学习很多有关迁移学习的内容。这是整个课程的重点。迁移学习研究的是如何使用一个已经能很好地完成一些任务的模型来完成新的任务。我们使用一个预训练的模型，然后调整它，不再使用ImageNet数据来预测一千种分类，而是使用宠物数据集来预测 37 种品种。这样的话，相对于一次常规的训练，你训练一个模型仅需百分之一的时间和数据，甚至更少。有可能会少于千分之一。记得我在展示的Nikhil的去年第一课项目的幻灯片吗，他只用了30张图片。ImageNet里没有板球和篮球的图片，但最终ImageNet还是很擅长识别世界里的各种事物，仅仅30个打篮球和板球的例子就足够构建一个几乎完美的分类器。 


## 过拟合 [[40:05](https://youtu.be/BWWm4AzsdLk?t=2405)]

等等，你怎么知道这个模型可以广泛识别出人们打板球和篮球的图片。或许它只是会识别出这30张图片，或许这只是作弊。这被称作“过拟合”。我们将在课程中讲解和多关于过拟合的内容。过拟合是指你的模型并没有学会识别图片，比如区分板球和篮球，而是仅仅能识别这几张特定图片里的板球运动员和这几张特定图片里的篮球运动员。我们必须确认我们没有过拟合。使用验证数据集可以检查有没有过拟合。验证集是你的模型没有使用过的一组图片。基于验证集的度量值(比如错误率)被自动地打印出来。当我们创建数据组时，它会自动创建一个验证集。我们将学习很多种创建和使用验证集地方法。因为我们尝试集成所有最佳实践，你几乎无法不使用验证集。因为如果你不使用验证集，你就不知道你是否过拟合。所以我们总是打印出验证集的度量。我们总是保证模型不接触到验证集。这些都是已经实现了的，这些方法都已经被集成在data bunch对象中。


## 拟合模型 [[41:40](https://youtu.be/BWWm4AzsdLk?t=2500)]
现在我们有了一个ConvLearner，我们可以开始拟合它。你可以使用一个叫`fit`的方法。但实践中，你应该总是使用`fit_one_cycle`这个方法。简单来讲，one cycle learning 是4月份发表的 [一篇论文](https://arxiv.org/pdf/1803.09820.pdf) ，它明显比以前的方法更快更准确。再重复一遍，我不想教大家怎样用2017年的方法做深度学习。在2018年，最好的拟合模型的方法是使用one cycle。 

现在我们遍历整个数据集四次，我们把数据展示给模型看四次来训练它。每次它看到一个图片，它会变得更好些。但这会花费时间，并且这意味着会过拟合。如果它看太多次同一个图片，它只会学会识别这个图片，而不是区分宠物品种。接下来的课程里我们将学习如何调整遍历次数，现在我们选择使用4次，来看看程序是怎样运行的，你可以看到，遍历4次后，错误率是6%，这花费了1分56秒。

```python
learn.fit_one_cycle(4)
```

```
Total time: 01:10
epoch  train loss  valid loss  error_rate
1      1.175709    0.318438    0.099800    (00:18)
2      0.492309    0.229078    0.075183    (00:17)
3      0.336315    0.211106    0.067199    (00:17)
4      0.233666    0.191813    0.057219    (00:17)
```
94%的情况下，我们可以在37个猫狗的品种中选择出正确的那个，我认为这个结果很不错。要衡量这个结果好到什么程度，或许我们应该回过头来看看论文。记住，我说过使用学术或者kaggle数据集的好处是我们可以把我们的方案和Kaggle或者学术界里的最好成绩做对比。宠物品种数据集最初出现在2012年，如果浏览这些文章，你在论文里找到一个实验的章节。在实验章节里，你可以找到准确率的部分。他们实现了很多不同的模型，就像你在文章里读到的一样，这些模型是专门针对宠物识别的，他们学习了宠物的头长什么样，身体长什么样，宠物的图形一般长什么样。然后把这些组合在一起。他们使用这些复杂的代码和算法得到了59%的准确率。所以在2012，这项专门针对宠物的分析得到了59%的准确率。他们是哈佛的顶尖研究者。现在，在2018，使用简单的三行代码，我们得到了94%的准确率 (也就是 6% 的错误率)。可见我们使用深度学习取得了多大的进步，以及，使用PyTorch和fastai，这是多么容易做到。


[[46:43](https://youtu.be/BWWm4AzsdLk?t=2803)]
我们仅仅训练了一个模型。我们还不是非常清楚这是如何做到的，但我们知道我们使用三四行代码，我们做到了远远超过2012年顶尖水平的准确率。对于识别不同品种的猫狗来说，6%的错误率听起来令人印象深刻。我们还不太了解它是怎样工作的，但我们将会学习这些。这就够了。

### 过去的学生最后悔的事:

![](/lesson1/102.png)



> ### **所以请运行代码。真正地去运行代码** [[47:54](https://youtu.be/BWWm4AzsdLk?t=2874)]



实践的最重要的技巧是学习和理解给程序输入了什么，程序输出了什么。 

![](/lesson1/103.png)

Fastai库很新，但它得到了很多关注。它让很多事情简单了很多，也让做一些新东西变得可能。真正理解fastai程序要花很不少精力。最好的方式是使用[fastai 文档](http://docs.fast.ai/).



### Keras[ [49:25](https://youtu.be/BWWm4AzsdLk?t=2965)]

![](/lesson1/105.png)

fastai和其他软件比起来怎么样?唯一的和fastai类似的致力于简化深度学习的主流软件是Keras。Keras是很棒的软件，在使用fastai之前，我们在之前的课程里使用Keras。它基于Tensorflow。之前它是简化深度学习软件的典范。现在，使用fastai做深度学习更容易。如果对比去年的猫狗大战的课程练习，fastai可以得到更高的准确率 (验证集上的错误率少于Keras的一半), 训练时间少于一半，代码行数是 1/6。代码行数比你认为的更重要，31行Keras代码意味着你要做大量的决定，设置很多参数，做很多设置。这些是为了获得最佳的结果你必须了解的。fastai的5行代码，尽可能多地为你做了这些设置。通常，fastai为你选择了最优的默认值。你将发现这是一个非常有用的库，不仅是在学习深度学习上，在发展深度学习上也有巨大作用。




[[50:53](https://youtu.be/BWWm4AzsdLk?t=3053)]

![](/lesson1/106.png)



How far can you take it? All of the research that we do at fastai uses the library and an example of the research we did which was recently featured in Wired describes a new breakthrough in a natural language processing which people are calling the ImageNet moment which is basically we broke a new state-of-the-art result in text classification which OpenAI then built on top of our paper with more computing, more data to do different tasks to take it even further. This is an example of something we've done in the last 6 months in conjunction with my colleague Sebastian Ruder - an example of something that's being built in the fastai library and you are going to learn how to use this brand new model in three lessons time. You're actually going to get this exact result from this exact paper yourself.


[[51:50](https://youtu.be/BWWm4AzsdLk?t=3110)]
![](/lesson1/107.png)
Another example, one of our alumni, Hamel Husain built a new system for natural language semantic code search, you can find it on Github where you can actually type in English sentences and find snippets of code that do the thing you asked for. Again, it's being built with the fastai library using the techniques you'll learn in the next seven weeks.



[[52:27](https://youtu.be/BWWm4AzsdLk?t=3147)]

The best place to learn about these things and get involved in these things is on the forums where as well as categories for each part of the course and there is also a general category for deep learning where people talk about research papers applications. 

Even though today, we are focusing on a small number of lines of code to a particular thing which is image classification and we are not learning much math or theory, over these seven weeks and then part two, we are going to go deeper and deeper. 

### Where can that take you? [[53:05](https://youtu.be/BWWm4AzsdLk?t=3185)]

![](/lesson1/108.png)

This is Sarah Hooker. She did our first course a couple of years ago. She started learning to code two years before she took our course. She started a nonprofit called Delta Analytics, they helped build this amazing system where they attached old mobile phones to trees in Kanyan rain forests and used it to listen for chainsaw noises, and then they used deep learning to figure out when there was a chainsaw being used and then they had a system setup to alert rangers to go out and stop illegal deforestation in the rainforests. That was something she was doing while she was in the course as part of her class projects. 

![](/lesson1/109.png)
She is now a Google Brain researcher, publishing some papers, and now she is going to Africa to set up a Google Brain's first deep learning research center in Africa. She worked her arse off. She really really invested in this course. Not just doing all of the assignments but also going out and reading Ian Goodfellow's book, and doing lots of other things. It really shows where somebody who has no computer science or math background at all can be now one of the world's top deep learning researchers and doing very valuable work.

[[54:49](https://youtu.be/BWWm4AzsdLk?t=3289)]

![](/lesson1/110.png)



Another example from our most recent course, Christine Payne. She is now at OpenAI and you can find [her post](http://christinemcleavey.com/clara-a-neural-net-music-generator/) and actually listen to her music samples of something she built to automatically create chamber music compositions. 

![](/lesson1/111.png)

She is a classical pianist. Now I will say she is not your average classical pianist. She's a classical pianist who also has a master's in medical research in Stanford, and studied neuroscience, and was a high-performance computing expert at DE Shaw, Co-Valedictorian at Princeton. Anyway. Very annoying person, good at everything she does. But I think it's really cool to see how a domain expert of playing piano can go through the fastai course and come out the other end as OpenAI fellow. 

Interestingly, one of our other alumni of the course recently interviewed her for a blog post series he is doing on top AI researchers and she said one of the most important pieces of advice she got was from me and she said the advice was:



> #### Pick one project. Do it really well. Make it fantastic. [56:20](https://youtu.be/BWWm4AzsdLk?t=3380)



We're going to be talking a lot about you doing projects and making them fantastic during this course.


[[56:36](https://youtu.be/BWWm4AzsdLk?t=3396)]
Having said that, I don't really want you to go to AI or Google Brain. What I really want you to do is to go back to your workplace or your passion project and apply these skills there. 

![](/lesson1/112.png)
MIT released a deep learning course and they highlighted in their announcement this medical imaging example. One of our students Alex who is a radiologist said you guys just showed a model overfitting. I can tell because I am a radiologist and this is not what this would look like on a chest film. This is what it should look like and as a deep learning practitioner, this is how I know this is what happened in your model. So Alex is combining his knowledge of radiology and his knowledge of deep learning to assess MIT's model from just two images very accurately. So this is actually what I want most of you to be doing is to take your domain expertise and combine it with the deep learning practical aspects you'll learn in this course and bring them together like Alex is doing here. So a lot of radiologists have actually gone through this course now and have built journal clubs and American Council of Radiology practice groups. There's a data science institute at the ACR now and Alex is one of the people who is providing a lot of leadership in this area. And I would love you to do the same kind of thing that Alex is doing which is to really bring deep learning leadership into your industry and to your social impact project, whatever it is that you are trying to do. 


[[58:22](https://youtu.be/BWWm4AzsdLk?t=3502)]

![](/lesson1/113.png)

Another great example. This is Melissa Fabros who is a English literature PhD who studied gendered language in English literature or something and actually Rachel at the previous job taught her to code. Then she came to the fastai course. She helped Kiva, a micro lending a social impact organization, to build a system that can recognize faces. Why is that necessary? We're going to be talking a lot about this but because most AI researchers are white men, most computer vision software can only recognize white male faces effectively. In fast, I think it was IBM system was like 99.8% accurate on common white face men versus 65% accurate on dark skinned women. So it's like 30 or 40 times worse for black women versus white men. This is really important because for Kiva, black women perhaps are the most common user base for their micro lending platform. So Melissa after taking our course, again working her arse off, and being super intense in her study and her work won this $1,000,000 AI challenge for her work for Kiva. 



[[59:53](https://youtu.be/BWWm4AzsdLk?t=3593)]

![](/lesson1/114.png)

Karthik did our course and realized that the thing he wanted to do wasn't at his company. It was something else which is to help blind people to understand the world around them. So he started a new startup called envision. You can download the app and point your phone to things and it will tell you what it sees. I actually talked to a blind lady about these kinds of apps the other day and she confirmed to me this is a super useful thing for visually disabled users.  




[[1:00:24](https://youtu.be/BWWm4AzsdLk?t=3624)]
![](/lesson1/115.png)

The level that you can get to, with the content that you're going to get over these seven weeks and with this software can get you right to the cutting edge in areas you might find surprising. I helped a team of some of our students and some collaborators on actually breaking the world record for how quickly you can train ImageNet. We used standard AWS cloud infrastructure, cost of $40 of compute to train this model using fastai library, the technique you learn in this course. So it can really take you a long way. So don't be put off by this what might seem pretty simple at first. We are going deeper and deeper.



[[1:01:17](https://youtu.be/BWWm4AzsdLk?t=3677)]
![](/lesson1/116.png)

You can also use it for other kinds of passion project. Helena Sarin - you should definitely check out her Twitter account [@glagolista](https://twitter.com/glagolista). This art is basically a new style of art that she's developed which combines her painting and drawing with generative adversarial models to create these extraordinary results. I think this is super cool. She is not a professional artists, she is a professional software developer but she keeps on producing these beautiful results. When she started, her art had not really been shown or discussed anywhere, now there's recently been some quite high profile article describing how she is creating a new form of art. 


![](/lesson1/117.png)

Equally important, Brad Kenstler who figured out how to make a picture of Kanye out of pictures of Patrick Stewart's head. Also something you will learn to do if you wish to. This particular type of what's called "style transfer" - it's a really interesting tweak that allowed him to do something that hadn't quite been done before. This particular picture helped him to get a job as a deep learning specialist at AWS.



[[1:02:41](https://youtu.be/BWWm4AzsdLk?t=3761)]

Another alumni actually worked at Splunk as a software engineer and he designed an algorithm which basically turned Splunk to be fantastically good at identifying fraud and we'll talk more about it shortly. 

![](/lesson1/118.png)

If you've seen Silicon Valley, the HBO series, the hotdog Not Hotdog app - that's actually a real app you can download and it was built by Tim Anglade as a fastai student project. So there's a lot of cool stuff that you can do. It was Emmy nominated. We only have one Emmy nominated fastai alumni at this stage, so please help change that.



![](/lesson1/119.png)
[[1:03:30](https://youtu.be/BWWm4AzsdLk?t=3810)]

The other thing, the forum thread can turn into these really cool things. So Francisco was a really boring McKinsey consultant like me. So Francisco and I both have this shameful past that we were McKinsey consultants, but we left and we're okay now. He started this thread saying like this stuff we've just been learning about building NLP in different languages, let's try and do lots of different languages, and he started this thing called the language model zoo and out of that, there's now been an academic competition was won in Polish that led to an academic paper, Thai state of the art, German state of the art, basically as students have been coming up with new state of the art results across lots of different languages and this all is entirely done by students working together through the forum. 

So please get on the forum. But don't be intimidated because everybody you see on the forum, the vast majority of posting post all the darn time. They've been doing this a lot and they do it a lot of the time. So at first, it can feel intimidating because it can feel like you're the only new person there. But you're not. You're all new people, so when you just get out there and say like "okay all you people getting these state of the art results in German language modeling, I can't start my server, I try to click the notebook and I get an error, what do I do?" People will help you. Just make sure you provide all the information ([how to ask for help](https://forums.fast.ai/t/how-to-ask-for-help/10421)). 

Or if you've got something to add! If people are talking about crop yield analysis and you're a farmer and you think oh I've got something to add, please mention it even if you are not sure it's exactly relevant. It's fine. Just get involved. Because remember, everybody else in the forum started out also intimidated. We all start out not knowing things. So just get out there and try it!


[[1:05:59](https://youtu.be/BWWm4AzsdLk?t=3959)]
**Question**: Why are we using ResNet as opposed to Inception?

There are lots of architectures to choose from and it would be fair to say there isn't one best one but if you look at things like the Stanford DAWNBench benchmark of image classification, you'll see in first place, second place,  third place, and fourth place all use ResNet. ResNet is good enough, so it's fine. 
![](/lesson1/120.png)

The main reason you might want a different architecture is if you want to do edge computing, so if you want to create a model that's going to sit on somebody's mobile phone. Having said that, even there, most of the time, I reckon the best way to get a model onto somebody's mobile phone is to run it on your server and then have your mobile phone app talk to it. It really makes life a lot easier and you get a lot more flexibility. But if you really do need to run something on a low powered device, then there are special architectures for that. So the particular question was about Inception. That's a particular another architecture which tends to be pretty memory intensive but it's okay. It's not terribly resilient. One of the things we try to show you is stuff which just tends to always work even if you don't quite tune everything perfectly. So ResNet tends to work pretty well across a wide range of different kind of details around choices that you might make. So I think it's pretty good.


[[1:07:58](https://youtu.be/BWWm4AzsdLk?t=4078)]

We've got this trained model and what's actually happened as we'll learn is it's basically creating a set of weights. If you've ever done anything like a linear regression or logistic regression, you'll be familiar with coefficients. We basically found some coefficients and parameters that work pretty well and it took us a minute and 56 seconds. So if we want to start doing some more playing around and come back later, we probably should save those weights. You can just go `learn.save` and give it a name. It's going to put it in a model subdirectory in the same place the data came from, so if you save different models or different data bunches from different datasets, they'll all be kept separate. So don't worry about it.

```python
learn.save('stage-1')
```



## Results [[1:08:54](https://youtu.be/BWWm4AzsdLk?t=4134)]

To see what comes out, we could use this class for class interpretation. We are going to use this factory method from learner, so we pass in a learn object. Remember a learn object knows two things: 
1. What's your data
2. What is your model. Now it's not just an architecture, it's actually a trained model 

That's all the information we need to interpret that model. 


```python
interp = ClassificationInterpretation.from_learner(learn)
```

One of the things, perhaps the most useful things to do is called plot_top_losses. We are going to be learning a lot about this idea of loss functions shortly but in short, a loss function is something that tells you how good was your prediction. Specifically that means if you predicted one class of cat with great confidence, but actually you were wrong, then that's going to have a high loss because you were very confident about the wrong answer. So that's what it basically means to have high loss. By plotting the top losses, we are going to find out what were the things that we were the most wrong on, or the most confident about what we got wrong. 


```python
interp.plot_top_losses(9, figsize=(15,11))

```
![](/lesson1/9.png)

It prints out four things. What do they mean? Perhaps we should look at the document.

We have already seen `help`, and `help` just prints out a quick little summary. But if you want to really see how to do something use `doc`.

![](/lesson1/121.png)



`doc` tells you the same information as `help` but it has this very important thing which is `Show in docs`. When you click on it, it pops up the documentation for that method or class or function or whatever:

![](/lesson1/122.png)

It starts out by showing us the same information about what are the parameters it takes a long with the doc string. But then tells you more information:

> The title of each image shows: prediction, actual, loss, probability of actual class.

The documentation always has working code. This is your friend when you're trying to figure out how to use these things. The other thing I'll mention is if you're somewhat experienced Python programmer, you'll find the source code of fastai really easy to read. We are trying to write everything in just a small number (much less than half a screen) of code. If you click on `[source]` you can jump straight to the source code.

![](/lesson1/123.png)

Here is plot_top_loss, and this is also a great way to find out how to use the fastai library. Because nearly every line of code here,  is calling stuff in the fastai library. So don't be afraid to look at the source code.


[[1:12:48](https://youtu.be/BWWm4AzsdLk?t=4368)]

So that's how we can look at top losses and these are perhaps the most important image classification interpretation tools that we have because it lets us see what we are getting wrong. In this case, if you are a dog and cat expert, you'll realize that the things that's getting wrong are breeds that are actually very difficult to tell apart and you'd be able to look at these and say "oh I can see why they've got this one wrong". So this is a really useful tool.


### Confusion matrix [1:13:21](https://youtu.be/BWWm4AzsdLk?t=4401)

Another useful tool, kind of, is to use something called a confusion matrix which basically shows you for every actual type of dog or cat, how many times was it predicted to be that dog or cat. But unfortunately, in this case, because it's so accurate, this diagonal basically says how it's pretty much right all the time. 
![](/lesson1/10.png)

And you can see there is slightly darker ones like a five here, it's really hard to read exactly what their combination is. So what I suggest you use is instead of, if you've got lots of classes, don't use confusion matrix, but this is my favorite named function in fastai and I'm very proud of this - you can call "most confused".

### Most confused [[1:13:52](https://youtu.be/BWWm4AzsdLk?t=4432)]

```python
interp.most_confused(min_val=2)
```
```
[('american_pit_bull_terrier', 'staffordshire_bull_terrier', 5),
 ('Birman', 'Ragdoll', 5),
 ('english_setter', 'english_cocker_spaniel', 4),
 ('staffordshire_bull_terrier', 'american_pit_bull_terrier', 4),
 ('boxer', 'american_bulldog', 4),
 ('Ragdoll', 'Birman', 3),
 ('miniature_pinscher', 'chihuahua', 3),
 ('Siamese', 'Birman', 3)]
```
`most_confused` will simply grab out of the confusion matrix the particular combinations of predicted and actual that got wrong the most often. So this case, `('american_pit_bull_terrier', 'staffordshire_bull_terrier', 7)`:
- Actual `'american_pit_bull_terrier'` 
- Prediction `'staffordshire_bull_terrier'`
- This particular combination happened 7 times.

So this is a very useful thing because you can look and say "with my domain expertise, does it make sense?"



### Unfreezing, fine-tuning, and learning rates [[1:14:38](https://youtu.be/BWWm4AzsdLk?t=4478)]

Let's make our model better. How? We can make it better by using fine-tuning. So far we fitted 4 epochs and it ran pretty quickly. The reason it ran pretty quickly is that there was a little trick we used. These convolutional networks, they have many layers. We'll learn a lot about exactly what layers are, but for now, just know it goes through a lot of computations. What we did was we added a few extra layers to the end and we only trained those. We basically left most of the model exactly as it was, so that's really fast. If we are trying to build a model at something that's similar to the original pre-trained model (in this case, similar to the ImageNet data), that works pretty well.

But what we really want to do is to go back and train the whole model. This is why we pretty much always use this two stage process. By default, when we call `fit` or `fit_one_cycle` on a ConvLearner, it'll just fine-tune these few extra layers added to the end and it will run very fast. It will basically never overfit but to really get it good, you have to call `unfreeze`. `unfreeze` is the thing that says please train the whole model. Then I can call fit_one_cycle again. 

```python
learn.unfreeze()
learn.fit_one_cycle(1)
```
```

Total time: 00:20
epoch  train_loss  valid_loss  error_rate
1      1.045145    0.505527    0.159681    (00:20)
```

Uh-oh. The error got much worse. Why? In order to understand why, we are actually going to have to learn more about exactly what's going on behind the scenes. So let's start out by trying to get an intuitive understanding of what's going on behind the scenes. We are going to do it by looking at pictures.


[[1:16:28](https://youtu.be/BWWm4AzsdLk?t=4588)]
![](/lesson1/100.png)

These pictures come from [a fantastic paper](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf) by Matt Zeiler who nowadays is a CEO of Clarify which is a very successful computer vision startup and his supervisor for his PhD Rob Fergus. They wrote a paper showing how you can visualize the layers of a convolutional neural network. A convolutional neural network, which we will learn mathematically about what the layers are shortly, but the basic idea is that your red, green, and blue pixel values that are numbers from nought to 255 go into the simple computation (i.e. the first layer) and something comes out of that, and then the result of that goes into a second layer, and the result of that goes into the third layer and so forth. There can be up to a thousand layers of neural network. ResNet34 has 34 layers, and ResNet50 has 50 layers, but let's look at layer one. There's this very simple computation which is a convolution if you know what they are. What comes out of this first layer? Well, we can actually visualize these specific coefficients, the specific parameters by drawing them as a picture. There's actually a few dozen of them in the first layer, so we don't draw all of them. Let's just look at 9 at random. 



[[1:17:45](https://youtu.be/BWWm4AzsdLk?t=4665)]

![](/lesson1/124.png)

Here are nine examples of the actual coefficients from the first layer. So these operate on groups of pixels that are next to each other. So this first one basically finds groups of pixels that have a little diagonal line, the second one finds diagonal line in the other direction, the third one finds gradients that go from yellow to blue, and so forth. They are very simple little filters. That's layer one of ImageNet pre-trained convolutional neural net. 

![](/lesson1/125.png)

Layer 2 takes the results of those filters and does a second layer of computation. The bottom right are nine examples of a way of visualizing one of the second layer features. AS you can see, it basically learned to create something that looks for top left corners. There are ones that learned to find right-hand curves, and little circles, etc. In layer one, we have things that can find just one line, and in layer 2, we can find things that have two lines joined up or one line repeated. If you then look over to the right, these nine show you nine examples of actual bits of the actual photos that activated this filter a lot. So in other words, the filter on the bottom right was good at finding these window corners etc. 

So this is the kind of stuff you've got to get a really good intuitive understanding for. The start of my neural net is going to find very simple gradients and lines, the second layer can find very simple shapes, the third layer can find  combination of those. 

![](/lesson1/126.png)

Now we can find repeating pattern of two dimensional objects or we can find things that joins together, or bits of text (although sometimes windows) - so it seems to find repeated horizontal patterns. There are also ones that seem to find edges of fluffy or flowery things or geometric patterns. So layer 3 was able to take all the stuff from layer 2 and combine them together.

![](/lesson1/127.png)

Layer 4 can take all the stuff from layer 3 and combine them together. By layer 4, we got something that can find dog faces or bird legs. 

By layer 5, we've got something that can find the eyeballs of bird and lizards, or faces of particular breeds of dogs and so forth. So you can see how by the time you get to layer 34, you can find specific dog breeds and cat breeds. This is kind of how it works.

So when we first trained (i.e. fine-tuned) the pre-trained model, we kept all of these layers that you've seen so far and we just trained a few more layers on top of all of those sophisticated features that are already being created. So now we are going back and saying "let's change all of these". We will start with where they are, but let's see if we can make them better. 

Now, it seems very unlikely that we can make layer 1 features better. It's very unlikely that the definition of a diagonal line is going to be different when we look at dog and cat breeds versus the ImageNet data that this was originally trained on. So we don't really want to change the layer 1 very much if at all. Or else, the last layers, like types of dog face seems very likely that we do want to change that. So you want this intuition, this understanding that the different layers of a neural network represents different level of semantic complexity. 


[[1:22:06](https://youtu.be/BWWm4AzsdLk?t=4926)]

This is why our attempt to fine-tune this model didn't work because by default, it trains all the layers at the same speed which is to say it will update those things representing diagonal lines and gradients just as much as it tries to update the things that represent the exact specifics of what an eyeball looks like, so we have to change that. 

To change it, we first of all need to go back to where we were before. We just broke this model, much worse than it started out. So if we just go:

```python
learn.load('stage-1')
```
This brings back the model that we saved earlier. So let's load that back up and now our models back to where it was before we killed it.

### Learning rate finder [[1:22:58](https://youtu.be/BWWm4AzsdLk?t=4978)]

Let's run learning rate finder. We are learning about what that is next week, but for now, just know this is the thing that figures out what is the fastest I can train this neural network at without making it zip off the rails and get blown apart. 


```python
learn.lr_find()
learn.recorder.plot()
```
![](/lesson1/11.png)

This will plot the result of our LR finder and what this basically shows you is this key parameter called a learning rate. The learning rate basically says how quickly am I updating the parameters in my model. The x-axis one here shows me what happens as I increase the learning rate. The y axis show what the loss is. So you can see, once the learning rate gets passed 10^-4, my loss gets worse. It actually so happens, in fact I can check this if I press <kbd>shift</kbd>+<kbd>tab</kbd> here, my learning defaults to 0.003. So you can see why our loss got worse. Because we are trying to fine-tune things now, we can't use such a high learning rate. So based on the learning rate finder, I tried to pick something well before it started getting worse. So I decided to pick `1e-6`. But there's no point training all the layers at that rate, because we know that the later layers worked just fine before when we were training much more quickly. So what we can actually do is we can pass a range of learning rates to `learn.fit_one_cycle`. And we do it like this:

```python
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
```
```
Total time: 00:41
epoch  train_loss  valid_loss  error_rate
1      0.226494    0.173675    0.057219    (00:20)
2      0.197376    0.170252    0.053227    (00:20)
```

You use this keyword in Python called `slice` and that can take a start value and a stop value and basically what this says is train the very first layers at a learning rate of 1e-6, and the very last layers at a rate of 1e-4, and distribute all the other layers across that (i.e. between those two values equally). 


### How to pick learning rates after unfreezing [[1:25:23](https://youtu.be/BWWm4AzsdLk?t=5123)]

A good rule of thumb is after you unfreeze (i.e. train the whole thing), pass a max learning rate parameter, pass it a slice, make the second part of that slice about 10 times smaller than your first stage. Our first stage defaulted to about 1e-3 so it's about 1e-4. And the first part of the slice should be a value from your learning rate finder which is well before things started getting worse. So you can see things are starting to get worse maybe about here:

![](/lesson1/128.png)

So I picked something that's at least 10 times smaller than that.

If I do that, then the error rate gets a bit better. So I would perhaps say for most people most of the time, these two stages are enough to get pretty much a world-class model. You won't win a Kaggle competition, particularly because now a lot of fastai alumni are competing on Kaggle and this is the first thing that they do. But in practice, you'll get something that's about as good in practice as the vast majority of practitioners can do. 

## ResNet50 [[1:26:55](https://youtu.be/BWWm4AzsdLk?t=5215)]

We can improve it by using more layers and we will do this next week but by basically doing a ResNet50 instead of ResNet34. And you can try running this during the week if you want to. You'll see it's exactly the same as before, but I'm using ResNet50. 

```python

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=320, bs=bs//2)
data.normalize(imagenet_stats)
```

```python

learn = ConvLearner(data, models.resnet50, metrics=error_rate)
```

What you'll find is it's very likely if you try to do this, you will get an error and the error will be your GPU has ran out of memory. The reason for that is that ResNet50 is bigger than ResNet34, and therefore, it has more parameters and use more of your graphics card memory, just totally separate to your normal computer RAM, this is GPU RAM. If you're using the default Salamander,  AWS, then you'll be having a 16G of GPU memory. The card I use most of the time has 11G GPU memory, the cheaper ones have 8G. That's kind of the main range you tend to get. If yours have less than 8G of GPU memory, it's going to be frustrating for you. 

It's very likely that if you try to run this, you'll get an out of memory error and that's because it's just trying to do too much - too many parameter updates for the amount of RAM you have. That's easily fixed. `ImageDataBunch` constructor has a parameter at the end `bs` - a batch size. This basically says how many images do you train at one time. If you run out of memory, just make it smaller.

It's fine to use a smaller bath size. It might take a little bit longer. That's all. So that's just one number you'll need to try during the week. 

```python
learn.fit_one_cycle(8, max_lr=slice(1e-3))
```
```
Total time: 07:08
epoch  train_loss  valid_loss  error_rate
1      0.926640    0.320040    0.076555    (00:52)
2      0.394781    0.205191    0.063568    (00:52)
3      0.307754    0.203281    0.069036    (00:53)
4      0.244182    0.160488    0.054682    (00:53)
5      0.185785    0.153520    0.049214    (00:53)
6      0.157732    0.149660    0.047163    (00:53)
7      0.107212    0.136898    0.043062    (00:53)
8      0.097324    0.136638    0.042379    (00:54)
```

Again, we fit it for a while and we get down to 4.2% error rage. So this is pretty extraordinary. I was pretty surprised because when we first did in the first course, this cats vs. dogs, we were getting somewhere around 3% error for something where you've got a 50% chance of being right and the two things look totally different. So the fact that we can get 4.2% error for such a fine grain thing, it's quite extraordinary. 

### Interpreting the results again [1:29:41](https://youtu.be/BWWm4AzsdLk?t=5381)

```python
interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)
```
```
[('Ragdoll', 'Birman', 7),
 ('american_pit_bull_terrier', 'staffordshire_bull_terrier', 6),
 ('Egyptian_Mau', 'Bengal', 6),
 ('Maine_Coon', 'Bengal', 3),
 ('staffordshire_bull_terrier', 'american_pit_bull_terrier', 3)]
```

You can call the most_confused here and you can see the kinds of things that it's getting wrong. Depending on when you run it, you're going to get slightly different numbers, but you'll get roughly the same kind of things. So quite often, I find the Ragdoll and Birman are things that it gets confused. I actually have never heard of either of those things, so I actually looked them up and found a page on the cat site called "Is this a Birman or Ragdoll kitten?" and there was a long thread of cat experts arguing intensely about which it is. So I feel fine that my computer had problems.   

![](/lesson1/129.png)

I found something similar, I think it was this pitbull versus staffordshire bull terrier, apparently the main difference is the particular kennel club guidelines as to how they are assessed. But some people thing that one of them might have a slightly redder nose. So this is the kind of stuff where actually even if you're not a domain expert, it helps you become one. Because I now know more about which kinds of pet breeds are hard to identify than I used to. So model interpretation works both ways. 

## Homework [[1:30:58](https://youtu.be/BWWm4AzsdLk?t=5458)]

So what I want you to do this week is to run this notebook, make sure you can get through it, but then I really want you to do is to get your own image dataset and actually Francisco is putting together a guide that will show you how to download data from Google Images so you can create your own dataset to play with. But before I go, I want to show you how to create labels in lots of different ways because your dataset where you get it from won't necessarily be that kind of regex based approach. It could be in lots of different formats. So to show you how to do this, I'm going to use the MNIST sample. MNIST is a picture of hand drawn numbers - just because I want to show you different ways of creating these datasets. 

```python
path = untar_data(URLs.MNIST_SAMPLE); path
```

```python
path.ls()
```
```
['train', 'valid', 'labels.csv', 'models']
```

You see there are a training set and the validation set already. So basically the people that put together this dataset have already decided what they want you to use as a validation set. 

### Scenario 1: Labels are folder names

```python
(path/'train').ls()
```
```
['3', '7']
```

There are a folder called 3 and a folder called 7. Now this is really common way to give people labels. Basically it says everything that's a three, I put in a folder called three. Everything that's a seven, I'll put in a folder called seven. This is often called an "ImageNet style dataset" because this is how ImageNet is distributed. So if you have something in this format where the labels are just whatever the folders are called, you can say `from_folder`.

```python
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26)
```

This will create an ImageDataBunch for you and as you can see it created the labels:

```python
data.show_batch(rows=3, figsize=(5,5))
```
![](/lesson1/12.png)


### Scenario 2: CSV file [[1:33:17](https://youtu.be/BWWm4AzsdLk?t=5597)]

Another possibility, and for this MNIST sample, I've got both, it might come with a CSV file that would look something like this.

```python
df = pd.read_csv(path/'labels.csv')
df.head()
```

![](/lesson1/130.png)

For each file name, what's its label. In this case, labels are not three or seven, they are 0 or 1 which basically is it a 7 or not. So that's another possibility. If this is how your labels are, you can use `from_csv`:

```python
data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28)
```

And if it is called `labels.csv`, you don't even have to pass in a file name. If it's called something else, then you can pass in the `csv_labels` 

```python
data.show_batch(rows=3, figsize=(5,5))
data.classes
```
```
[0, 1]
```

![](/lesson1/13.png)



### Scenario 3: Using regular expression

```python
fn_paths = [path/name for name in df['name']]; fn_paths[:2]
```

```
[PosixPath('/home/jhoward/.fastai/data/mnist_sample/train/3/7463.png'),
 PosixPath('/home/jhoward/.fastai/data/mnist_sample/train/3/21102.png')]
```

This is the same thing, these are the folders. But I could actually grab the label by using a regular expression. We've already seen this approach:


```python
pat = r"/(\d)/\d+\.png$"
data = ImageDataBunch.from_name_re(path, fn_paths, pat=pat, ds_tfms=tfms, size=24)
data.classes
```

```
['3', '7']
```



### Scenario 4: Something more complex [[1:34:21](https://youtu.be/BWWm4AzsdLk?t=5661)]

You can create an arbitrary function that extracts a label from the file name or path. In that case, you would say `from_name_func`:

```python
data = ImageDataBunch.from_name_func(path, fn_paths, ds_tfms=tfms, size=24,
        label_func = lambda x: '3' if '/3/' in str(x) else '7')
data.classes
```



### Scenario 5: You need something even more flexible

If you need something even more flexible than that, you're going to write some code to create an array of labels. So in that case, you can just use `from_lists` and pass in the array.

```python
labels = [('3' if '/3/' in str(x) else '7') for x in fn_paths]
labels[:5]
```

```python
data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=24)
data.classes
```

So you can see there's lots of different ways of creating labels. So during the week, try this out.

Now you might be wondering how would you know to do all these things? Where am I going to find this kind of information? So I'll show you something incredibly cool. You know how to get documentation:

```python
doc(ImageDataBunch.from_name_re)
```

[[Show in docs](https://docs.fast.ai/vision.data.html#ImageDataBunch.from_name_re)]



![](/lesson1/131.png)



Every single line of code I just showed you, I took it this morning and I copied and pasted it from the documentation. So you can see here the exact code that I just used. So the documentation for fastai doesn't just tell you what to do, but step to step how to do it. And here is perhaps the coolest bit. If you go to [fastai/fastai_docs](https://github.com/fastai/fastai_docs) and click on [docs/src](https://github.com/fastai/fastai_docs/tree/master/docs_src).

All of our documentation is actually just Jupyter Notebooks. You can git clone this repo and if you run it, you can actually run every single line of the documentation yourself.

This is the kind of the ultimate example to me of experimenting. Anything that you read about in the documentation, nearly everything in the documentation has actual working examples in it with actual datasets that are already sitting in there in the repo for you. So you can actually try every single function in your browser, try seeing what goes in and try seeing what comes out. 


[[1:37:27](https://youtu.be/BWWm4AzsdLk?t=5847)]

**Question**: Will the library use multi GPUs in parallel by default? 

The library will use multiple CPUs by default but just one GPU by default. We probably won't be looking at multi GPU until part 2. It's easy to do and you'll find it on the forum, but most people won't be needing to use that now.

**Question**: Can the library use 3D data such as MRI or CAT scan?

Yes, it can. ANd there is actually a forum thread about that already. Although that's not as developed as 2D yet but maybe by the time the MOOC is out, it will be.



### Splunk Anti-Fraud Software [[1:38:10](https://youtu.be/BWWm4AzsdLk?t=5890)]

[blog](https://www.splunk.com/blog/2017/04/18/deep-learning-with-splunk-and-tensorflow-for-security-catching-the-fraudster-in-neural-networks-with-behavioral-biometrics.html)

Before I wrap up, I'll just show you an example of the kind of interesting stuff that you can do by doing this kind of exercise. 

Remember earlier I mentioned that one of our alumni who works at Splunk which is a NASDAQ listed big successful company created this new anti-fraud software. This is actually how he created it as part of a fastai part 1 class project:

![](/lesson1/132.jpg)


He took the telemetry of users who had Splunk analytics installed and watched their mouse movements and he created pictures of the mouse movements. He converted speed into color and right and left clicks into splotches. He then took the exact code that we saw with an earlier version of the software and trained a CNN in exactly the same way we saw and used that to train his fraud model. So he took something which is not obviously a picture and he turned it into a picture and got these fantastically good results for a piece of fraud analysis software. 

So it pays to think creatively. So if you are wanting to study sounds, a lot of people that study sounds do it by actually creating a spectrogram image and then sticking that into a ConvNet. So there's a lot of cool stuff you can do with this. 

So during the week, get your GPU going, try and use your first notebook, make sure that you can use lesson 1 and work through it. Then see if you can repeat the process on your own dataset. Get on the forum and tell us any little success you had. Any constraints you hit, try it for an hour or two but if you get stuck, please ask. If you are able to successfully build a model with a new dataset, let us know! I will see you next week.
