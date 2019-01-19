# Lesson 7

[Video](https://youtu.be/nWpdkZE2_cc) / [Course Forum](https://forums.fast.ai/t/lesson-7-official-resources/32553)

Welcome to lesson 7! The last lesson of part 1. This will be a pretty intense lesson. Don't let that bother you because partly what I want to do is to give you enough things to think about to keep you busy until part 2. In fact, some of the things we cover today, I'm not going to tell you about some of the details. I'll just point out a few things. I'll say like okay that we're not talking about yet, that we're not talking about yet. Then come back in part 2 to get the details on some of these extra pieces. So today will be a lot of material pretty quickly. You might require a few viewings to fully understand at all or a few experiments and so forth. That's kind of intentional. I'm trying to give you stuff to to keep you amused for a couple of months.

![](lesson7/1.png)

I wanted to start by showing some cool work done by a couple of students; Reshama and Nidhin who have developed an Android and an iOS app, so check out [Reshma's post on the forum](https://forums.fast.ai/t/share-your-work-here/27676/679?u=hiromi) about that because they have a demonstration of how to create both Android and iOS apps that are actually on the Play Store and on the Apple App Store, so that's pretty cool. First ones I know of that are on the App Store's that are using fast.ai. Let me also say a huge thank you to Reshama for all of the work she does both for the fast.ai community and the machine learning community more generally, and also the [Women in Machine Learning](https://wimlworkshop.org/) community in particular. She does a lot of fantastic work including providing lots of fantastic documentation and tutorials and community organizing and so many other things. So thank you, Reshama and congrats on getting this app out there.

## MNIST CNN [[2:04](https://youtu.be/nWpdkZE2_cc?t=124)]

We have lots of lesson 7 notebooks today, as you see. The first notebook we're going to look at is [lesson7-resnet-mnist.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-resnet-mnist.ipynb). What I want to do is look at some of the stuff we started talking about last week around convolutions and convolutional neural networks, and start building on top of them to create a fairly modern deep learning architecture largely from scratch. When I say from scratch, I'm not going to re-implement things we already know how to implement, but use the pre-existing PyTorch bits of those. So we're going to use the MNIST dataset. `URLs.MNIST` has the whole MNIST dataset, often we've done stuff with a subset of it. 

```python
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```

```python
from fastai.vision import *
```

```python
path = untar_data(URLs.MNIST)
```

```python
path.ls()
```

```
[PosixPath('/home/jhoward/.fastai/data/mnist_png/training'),
 PosixPath('/home/jhoward/.fastai/data/mnist_png/models'),
 PosixPath('/home/jhoward/.fastai/data/mnist_png/testing')]
```

In there, there's a training folder and a testing folder. As I read this in, I'm going to show some more details about pieces of the data blocks API, so that you see what's going on.  Normally with the date blocks API, we've kind of said `blah.blah.blah.blah.blah` and done it all in one cell, but let's do it in one cell at a time.

```python
il = ImageItemList.from_folder(path, convert_mode='L')
```

First thing you say is what kind of item list do you have. So in this case it's an item list of images. Then where are you getting the list of file names from. In this case, by looking in a folder recursively. That's where it's coming from. 

You can pass in arguments that end up going to Pillow because Pillow (a.k.a. PIL) is the thing that actually opens that for us, and in this case these are black and white rather than RGB, so you have to use Pillow's `convert_mode='L'`. For more details refer to the python imaging library documentation to see what their convert modes are. But this one is going to be a grayscale which is what MNIST is.

```python
il.items[0]
```

```
PosixPath('/home/jhoward/.fastai/data/mnist_png/training/8/56315.png')
```

So inside an item list is an `items` attribute, and the `items` attribute is kind of thing that you gave it. It's the thing that it's going to use to create your items. So in this case, the thing you gave it really is a list of file names. That's what it got from the folder. 

```python
defaults.cmap='binary'
```

When you show images, normally it shows them in RGB. In this case, we want to use a binary color map. In fast.ai, you can set a default color map. For more information about cmap and color maps, refer to the matplotlib documentation. And `defaults.cmap='binary'` world set the default color map for fast.ai.

```python
il
```

```
ImageItemList (70000 items)
[Image (1, 28, 28), Image (1, 28, 28), Image (1, 28, 28), Image (1, 28, 28), Image (1, 28, 28)]...
Path: /home/jhoward/.fastai/data/mnist_png
```

Our image item list contains 70,000 items, and it's a bunch of images that are 1 by 28 by 28. Remember that PyTorch puts channel first, so they are one channel 28x28. You might think why aren't there just 28 by 28 matrices rather than a 1 by 28 by 28 rank 3 tensor. It's just easier that way. All the `Conv2d` stuff and so forth works on rank 3 tensors, so you want to include that unit axis at the start, so fast.ai will do that for you even when it's reading one channel images.

```python
il[0].show()
```

![](lesson7/2.png)

The `.items` attribute contains the things that's read to build the image which in this case is the file name, but if you just index into an item list directly, you'll get the actual image object. The actual image object has a `show` method, and so there's the image.

```python
sd = il.split_by_folder(train='training', valid='testing')
```

Once you've got an image item list, you then split it into training versus validation. You nearly always want validation. If you don't, you can actually use the `.no_split` method to create an empty validation set. You can't skip it entirely. You have to say how to split, and one of the options is `no_split`. 

So remember, that's always the order. First create your item list, then decide how to split. In this case, we're going to do it based on folders. The validation folder for MNIST is called `testing`. In fast.ai parlance, we use the same kind of parlance that Kaggle does which is the training set is what you train on, the validation set has labels and you do it for testing that your models working. The test set doesn't have labels and you use it for doing inference, submitting to a competition, or sending it off to somebody who's held out those labels for vendor testing or whatever. So just because a folder in your data set is called `testing`, doesn't mean it's a test set. This one has labels, so it's a validation set.

If you want to do inference on lots of things at a time rather than one thing at a time, you want to use the `test=` in fast.ai to say this is stuff which has no labels and I'm just using for inference.

[[6:54](https://youtu.be/nWpdkZE2_cc?t=414)]

```python
sd
```

```
ItemLists;

Train: ImageItemList (60000 items)
[Image (1, 28, 28), Image (1, 28, 28), Image (1, 28, 28), Image (1, 28, 28), Image (1, 28, 28)]...
Path: /home/jhoward/.fastai/data/mnist_png;

Valid: ImageItemList (10000 items)
[Image (1, 28, 28), Image (1, 28, 28), Image (1, 28, 28), Image (1, 28, 28), Image (1, 28, 28)]...
Path: /home/jhoward/.fastai/data/mnist_png;

Test: None
```

So my split data is a training set and a validation set, as you can see.



```
(path/'training').ls()
```

```
[PosixPath('/home/jhoward/.fastai/data/mnist_png/training/8'),
 PosixPath('/home/jhoward/.fastai/data/mnist_png/training/5'),
 PosixPath('/home/jhoward/.fastai/data/mnist_png/training/2'),
 PosixPath('/home/jhoward/.fastai/data/mnist_png/training/3'),
 PosixPath('/home/jhoward/.fastai/data/mnist_png/training/9'),
 PosixPath('/home/jhoward/.fastai/data/mnist_png/training/6'),
 PosixPath('/home/jhoward/.fastai/data/mnist_png/training/1'),
 PosixPath('/home/jhoward/.fastai/data/mnist_png/training/4'),
 PosixPath('/home/jhoward/.fastai/data/mnist_png/training/7'),
 PosixPath('/home/jhoward/.fastai/data/mnist_png/training/0')]
```

Inside the training set, there's a folder for each class. 

```python
ll = sd.label_from_folder()
```

Now we can take that split data and say `label_from_folder`.

So first you create the item list, then you split it, then you label it.

```python
ll
```

```
LabelLists;

Train: LabelList
y: CategoryList (60000 items)
[Category 8, Category 8, Category 8, Category 8, Category 8]...
Path: /home/jhoward/.fastai/data/mnist_png
x: ImageItemList (60000 items)
[Image (1, 28, 28), Image (1, 28, 28), Image (1, 28, 28), Image (1, 28, 28), Image (1, 28, 28)]...
Path: /home/jhoward/.fastai/data/mnist_png;

Valid: LabelList
y: CategoryList (10000 items)
[Category 8, Category 8, Category 8, Category 8, Category 8]...
Path: /home/jhoward/.fastai/data/mnist_png
x: ImageItemList (10000 items)
[Image (1, 28, 28), Image (1, 28, 28), Image (1, 28, 28), Image (1, 28, 28), Image (1, 28, 28)]...
Path: /home/jhoward/.fastai/data/mnist_png;

Test: None
```

You can see now we have an `x` and the `y`, and the `y` are category objects. Category object is just a class basically. 

```python
x,y = ll.train[0]
```

If you index into a label list such as `ll.train` as a label list, you will get back an independent variable and independent variable (i.e. x and y). In this case, the `x` will be an image object which I can show, and the `y` will be a category object which I can print:

```
x.show()
print(y,x.shape)
```

```
8 torch.Size([1, 28, 28])
```

![](lesson7/2.png)

That's the number 8 category, and there's the 8.

[[7:56](https://youtu.be/nWpdkZE2_cc?t=476)]

```python
tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])
```

Next thing we can do is to add transforms. In this case, we're not going to use the normal `get_transforms` function because we're doing digit recognition and digit recognition, you wouldn't want to flip it left right. That would change the meaning of it. You wouldn't want to rotate it too much, that would change the meaning of it. Also because these images are so small, doing zooms and stuff is going to make them so fuzzy as to be unreadable. So normally, for small images of digits like this, you just add a bit of random padding. So I'll use the random padding function which actually returns two transforms; the bit that does the padding and the bit that does the random crop. So you have to use star(`*`) to say put both these transforms in this list.

```python
ll = ll.transform(tfms)
```

Now we call transform. This empty array here is referring to the validation set transforms:

![](lesson7/3.png)

So no transforms with the validation set.

Now we've got a transformed labeled list, we can pick a batch size and choose data bunch:

```python
bs = 128
```

```python
# not using imagenet_stats because not using pretrained model
data = ll.databunch(bs=bs).normalize()
```

We can choose normalize. In this case, we're not using a pre-trained model, so there's no reason to use ImageNet stats here. So if you call normalize like this without passing in stats, it will grab a batch of data at random and use that to decide what normalization stats to use. That's a good idea if you're not using a pre-trained model.

```python
x.show()
print(y)
```

```
8
```

![](lesson7/2.png)

Okay, so we've got a data bunch and in that data bunch is a data set which we've seen already. But what is interesting is that the training data set now has data augmentation because we've got transforms. `plot_multi` is a fast.ai function that will plot the result of calling some function for each of this row by column grid. So in this case, my function is just grab the first image from the training set and because each time you grab something from the training set, it's going to load it from disk and it's going to transform it on the fly. People sometimes ask how many transformed versions of the image do you create and the answer is infinite. Each time we grab one thing from the data set, we do a random transform on the fly, so potentially every one will look a little bit different. So you can see here, if we plot the result of that lots of times, we get 8's in slightly different positions because we did random padding.

```
def _plot(i,j,ax): data.train_ds[0][0].show(ax, cmap='gray')
plot_multi(_plot, 3, 3, figsize=(8,8))
```

![](lesson7/4.png)

[[10:27](https://youtu.be/nWpdkZE2_cc?t=627)]

You can always grab a batch of data then from the data bunch, because remember, data bunch has data loaders, and data loaders are things you grab a batch at a time. So you can then grab a X batch and a Y batch, look at their shape - batch size by channel by row by column:

```python
xb,yb = data.one_batch()
xb.shape,yb.shape
```

```
(torch.Size([128, 1, 28, 28]), torch.Size([128]))
```

All fast.ai data bunches have a show_batch which will show you what's in it in some sensible way:

```python
data.show_batch(rows=3, figsize=(5,5))
```

![](lesson7/5.png)



That was a quick walk through with a data block API stuff to grab our data.

### Basic CNN with batch norm [11:01](https://youtu.be/nWpdkZE2_cc?t=661)

Let's start out creating a simple CNN. The input is 28 by 28. I like to define when I'm creating architectures a function which kind of does the things that I do again and again and again. I don't want to call it with the same arguments because I'll forget or I make a mistake. In this case, all of my convolution is going to be kernel size 3 stride 2 padding 1. So let's just create a simple function to do a conv with those parameters:

```python
def conv(ni,nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)
```

Each time you have a convolution, it's skipping over one pixel so it's jumping two steps each time. That means that each time we have a convolution, it's going to halve the grid size. I've put a comment here showing what the new grid size is after each one.

```python
model = nn.Sequential(
    conv(1, 8), # 14
    nn.BatchNorm2d(8),
    nn.ReLU(),
    conv(8, 16), # 7
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 32), # 4
    nn.BatchNorm2d(32),
    nn.ReLU(),
    conv(32, 16), # 2
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 10), # 1
    nn.BatchNorm2d(10),
    Flatten()     # remove (1,1) grid
)
```

After the first convolution, we have one channel coming in because it's a grayscale image with one channel, and then how many channels coming out? Whatever you like. So remember, you always get to pick how many filters you create regardless of whether it's a fully connected layer in which case it's just the width of the matrix you're multiplying by, or in this case with the 2D conv, it's just how many filters do you want. So I picked 8 and so after this, it's stride 2 to so the 28 by 28 image is now a 14 by 14 feature map with 8 channels. Specifically therefore, it's an 8 by 14 by 14 tensor of activations.

Then we'll do a batch norm, then we'll do ReLU. The number of input filters to the next conv has to equal the number of output filters from the previous conv, and we can just keep increasing the number of channels because we're doing stride 2, it's got to keep decreasing the grid size. Notice here, it goes from 7 to 4 because if you're doing a stride 2 conv over 7, it's going to be  `math.ceiling` of 7/2. 

Batch norm, ReLU, conv. We are now down to 2 by 2. Batch norm, ReLU, conv, we're now down to 1 by 1. After this, we have a feature map of 10 by 1 by 1. Does that make sense? We've got a grid size of one now. It's not a vector of length 10, it's a rank 3 tensor of 10 by 1 by 1. Our loss functions expect (generally) a vector not a rank 3 tensor, so you can chuck `flatten` at the end, and flatten just means remove any unit axes. So that will make it now just a vector of length 10 which is what we always expect.

That's how we can create a CNN. Then we can return that into a learner by passing in the data and the model and the loss function and optionally some metrics. We're going to use cross-entropy as usual. We can then call `learn.summary()` and confirm.

```python
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
```

```python
learn.summary()
```

```
================================================================================
Layer (type)               Output Shape         Param #   
================================================================================
Conv2d                    [128, 8, 14, 14]     80                  
________________________________________________________________________________
BatchNorm2d               [128, 8, 14, 14]     16                  
________________________________________________________________________________
ReLU                      [128, 8, 14, 14]     0                   
________________________________________________________________________________
Conv2d                    [128, 16, 7, 7]      1168                
________________________________________________________________________________
BatchNorm2d               [128, 16, 7, 7]      32                  
________________________________________________________________________________
ReLU                      [128, 16, 7, 7]      0                   
________________________________________________________________________________
Conv2d                    [128, 32, 4, 4]      4640                
________________________________________________________________________________
BatchNorm2d               [128, 32, 4, 4]      64                  
________________________________________________________________________________
ReLU                      [128, 32, 4, 4]      0                   
________________________________________________________________________________
Conv2d                    [128, 16, 2, 2]      4624                
________________________________________________________________________________
BatchNorm2d               [128, 16, 2, 2]      32                  
________________________________________________________________________________
ReLU                      [128, 16, 2, 2]      0                   
________________________________________________________________________________
Conv2d                    [128, 10, 1, 1]      1450                
________________________________________________________________________________
BatchNorm2d               [128, 10, 1, 1]      20                  
________________________________________________________________________________
Lambda                    [128, 10]            0                   
________________________________________________________________________________
Total params:  12126
```

After that first conv, we're down to 14 by 14 and after the second conv 7 by 7, 4 by 4, 2 by 2, 1 by 1. The `flatten` comes out (calling it a `Lambda`), that as you can see it gets rid of the 1 by 1 and it's now just a length 10 vector for each item in the batch so 128 by 10 matrix in the whole mini batch.

Just to confirm that this is working okay, we can grab that mini batch of X that we created earlier (there's a mini batch of X), pop it onto the GPU, and call the model directly. Any PyTorch module, we can pretend it's a function and that gives us back as we hoped a 128 by 10 result.

```python
xb = xb.cuda()
```

```python
model(xb).shape
```

```
torch.Size([128, 10])
```

That's how you can directly get some predictions out. LR find, fit one cycle, and bang. We already have a 98.6% accurate conv net. 

```python
learn.lr_find(end_lr=100)
```

```python
learn.recorder.plot()
```

![](lesson7/6.png)

```python
learn.fit_one_cycle(3, max_lr=0.1)
```

Total time: 00:18

| epoch | train_loss | valid_loss | accuracy |
| ----- | ---------- | ---------- | -------- |
| 1     | 0.215413   | 0.169024   | 0.945300 |
| 2     | 0.129223   | 0.080600   | 0.974500 |
| 3     | 0.071847   | 0.042908   | 0.986400 |

This is trained from scratch, of course, it's not pre-trained. We've literally created our own architecture. It's about the simplest possible architecture you can imagine. 18 seconds to train, so that's how easy it is to create a pretty accurate digit detector.

### Refactor [15:42](https://youtu.be/nWpdkZE2_cc?t=942)

Let's refactor that a little. Rather than saying conv, batch norm, ReLU all the time, fast.ai already has something called `conv_layer` which lets you create conv, batch norm, ReLU combinations. It has various other options to do other tweaks to it, but the basic version is just exactly what I just showed you. So we can refactor that like so:

```python
def conv2(ni,nf): return conv_layer(ni,nf,stride=2)
```

```python
model = nn.Sequential(
    conv2(1, 8),   # 14
    conv2(8, 16),  # 7
    conv2(16, 32), # 4
    conv2(32, 16), # 2
    conv2(16, 10), # 1
    Flatten()      # remove (1,1) grid
)
```

That's exactly the same neural net.

```python
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
```

```python
learn.fit_one_cycle(10, max_lr=0.1)
```

Total time: 00:53

| epoch | train_loss | valid_loss | accuracy |
| ----- | ---------- | ---------- | -------- |
| 1     | 0.222127   | 0.147457   | 0.955700 |
| 2     | 0.189791   | 0.305912   | 0.895600 |
| 3     | 0.167649   | 0.098644   | 0.969200 |
| 4     | 0.134699   | 0.110108   | 0.961800 |
| 5     | 0.119567   | 0.139970   | 0.955700 |
| 6     | 0.104864   | 0.070549   | 0.978500 |
| 7     | 0.082227   | 0.064342   | 0.979300 |
| 8     | 0.060774   | 0.055740   | 0.983600 |
| 9     | 0.054005   | 0.029653   | 0.990900 |
| 10    | 0.050926   | 0.028379   | 0.991100 |

Let's just try a little bit longer and it's actually 99.1% accurate if we train it for all of a minute, so that's cool.

### ResNet-ish [16:24](https://youtu.be/nWpdkZE2_cc?t=984)

How can we improve this? What we really want to do is create a deeper network, and so a very easy way to create a deeper network would be after every stride 2 conv, add a stride 1 conv. Because the stride 1 conv doesn't change the feature map size at all, so you can add as many as you like. But there's a problem. The problem was pointed out in this paper, very very very influential paper, called [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He and colleagues at (then) Microsoft Research.

They did something interesting. They said let's look at the training error. So forget generalization even, let's just look at the training error of a network trained on CIFAR-10 and let's try one network of 20 layers just basic 3x3 convs - basically the same network I just showed you, but without batch norm. They trained a 20 layer one and a 56 layer one on the training set. 

The 56 layer one has a lot more parameters. It's got a lot more of these stride 1 convs in the middle. So the one with more parameters should seriously over fit, right? So you would expect the 56 layer one to zip down to zero-ish training error pretty quickly and that is not what happens. It is worse than the shallower network.

![](lesson7/7.png)

When you see something weird happen, really good researchers don't go "oh no, it's not working" they go "that's interesting." So Kaiming He said "that's interesting. What's going on?" and he said "I don't know, but what I do know is this - I could take this 56 layer network and make a new version of it which is identical but has to be at least as good as the 20 layer network and here's how:

![](lesson7/8.png)

Every to convolutions, I'm going to add together the input to those two convolutions with the result of those two convolutions." In other words, he's saying instead of saying:

<img src="https://latex.codecogs.com/gif.latex?Output=Conv2(Conv1(x))" title="Output=Conv2(Conv1(x))" />

Instead, he's saying:

<img src="https://latex.codecogs.com/gif.latex?Output=x&plus;Conv2(Conv1(x))" title="Output=x+Conv2(Conv1(x))" />

His theory was 56 layers worth of convolutions in that has to be at least good as the 20 layer version because it could always just set conv2 and conv1 to a bunch of 0 weights for everything except for the first 20 layers because the X (i.e. the input) could just go straight through. So this thing here is (as you see) called an **identity connection**. It's the identity function - nothing happens at all. It's also known as a **skip connection**. 

So that was the theory. That's what the paper describes as the intuition behind this is what would happen if we created something which has to train at least as well as a 20 layer neural network because it kind of contains that 20 layer neural network. There's literally a path you can just skip over all the convolutions. So what happens? 

What happened was he won ImageNet that year. He easily won ImageNet that year. In fact, even today, we had that record-breaking result on ImageNet speed training ourselves in the last year, we used this too. ResNet has been revolutionary. 

### ResBlock Trick [20:36](https://youtu.be/nWpdkZE2_cc?t=1236)

Here's a trick if you're interested in doing some research. Anytime you find some model for anything whether it's medical image segmentation or some kind of GAN or whatever and it was written a couple of years ago, they might have forgotten to put ResBlocks in. Figure 2 is what we normally call a ResBlock. They might have forgotten to put ResBlocks in. So replace their convolutional path with a bunch of ResBlocks and you will almost always get better results faster. It's a good trick.

#### [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913) [[21:16](https://youtu.be/nWpdkZE2_cc?t=1276)]

At NeurIPS, which Rachel, I, David, and Sylvain all just came back from, we saw a new presentation where they actually figured out how to visualize the loss surface of a neural net which is really cool. This is a fantastic paper and anybody who's watching this lesson 7 is at a point where they will understand the most of the important concepts in this paper. You can read this now. You won't necessarily get all of it, but I'm sure you'll get it enough to find it interesting.

![](lesson7/9.png)

The big picture was this one. Here's what happens if you if you draw a picture where x and y here are two projections of the weight space, and z is the loss. As you move through the weight space, a 56 layer neural network without skip connections is very very bumpy. That's why this got nowhere because it just got stuck in all these hills and valleys. The exact same network with identity connections (i.e. with skip connections) has this loss landscape (on the right). So it's kind of interesting how Kaiming He recognized back in 2015 this shouldn't happen, here's a way that must fix it and it took three years before people were able to say oh this is kind of why it fixed it. It kind of reminds me of the batch norm discussion we had a couple of weeks ago that people realizing a little bit after the fact sometimes what's going on and why it helps.

```python
class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf,nf)
        self.conv2 = conv_layer(nf,nf)
        
    def forward(self, x): return x + self.conv2(self.conv1(x))
```



In our code, we can create a ResBlock in just the way I described. We create a `nn.Module`, we create two conv layers (remember, a `conv_layer` is Conv2d, ReLU, batch norm), so create two of those and then in forward we go `conv1(x)`, `conv2` of that and then add `x`.

```python
help(res_block)
```

```
Help on function res_block in module fastai.layers:

res_block(nf, dense:bool=False, norm_type:Union[fastai.layers.NormType, NoneType]=<NormType.Batch: 1>, bottle:bool=False, **kwargs)
    Resnet block of `nf` features.
```

There's a `res_block` function already in fast.ai so you can just call `res_block` instead, and you just pass in something saying how many filters you want.

```python
model = nn.Sequential(
    conv2(1, 8),
    res_block(8),
    conv2(8, 16),
    res_block(16),
    conv2(16, 32),
    res_block(32),
    conv2(32, 16),
    res_block(16),
    conv2(16, 10),
    Flatten()
)
```

There's the ResBlock that I defined in our notebook, and so with that ResBlock, I've just copied the previous CNN and after every conv2 except the last one, I added a res_block so this has now got three times as many layers, so it should be able to do more compute. But it shouldn't be any harder to optimize. 

Let's just refactor it one more time. Since I go `conv2` `res_block` so many times, let's just pop that into a little mini sequential model here and so I can refactor that like so:

```python
def conv_and_res(ni,nf): return nn.Sequential(conv2(ni, nf), res_block(nf))
```

```python
model = nn.Sequential(
    conv_and_res(1, 8),
    conv_and_res(8, 16),
    conv_and_res(16, 32),
    conv_and_res(32, 16),
    conv2(16, 10),
    Flatten()
)
```

Keep refactoring your architectures if you're trying novel architectures because you'll make less mistakes. Very few people do this. Most research code you look at is clunky as all heck and people often make mistakes in that way, so don't do that. You're all coders, so use your coding skills to make life easier. 

[[24:47](https://youtu.be/nWpdkZE2_cc?t=1487)]

Okay, so there's my ResNet-ish architecture. `lr_find` as usual, `fit` for a while, and I get 99.54%. 

```python
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
```

```python
learn.lr_find(end_lr=100)
learn.recorder.plot()
```

![](lesson7/10.png)

```python
learn.fit_one_cycle(12, max_lr=0.05)
```

Total time: 01:48

| epoch | train_loss | valid_loss | accuracy |
| ----- | ---------- | ---------- | -------- |
| 1     | 0.179228   | 0.102691   | 0.971300 |
| 2     | 0.111155   | 0.089420   | 0.973400 |
| 3     | 0.099729   | 0.053458   | 0.982500 |
| 4     | 0.085445   | 0.160019   | 0.951700 |
| 5     | 0.074078   | 0.063749   | 0.980800 |
| 6     | 0.057730   | 0.090142   | 0.973800 |
| 7     | 0.054202   | 0.034091   | 0.989200 |
| 8     | 0.043408   | 0.042037   | 0.986000 |
| 9     | 0.033529   | 0.023126   | 0.992800 |
| 10    | 0.023253   | 0.017727   | 0.994400 |
| 11    | 0.019803   | 0.016165   | 0.994900 |
| 12    | 0.023228   | 0.015396   | 0.995400 |

That's interesting because we've trained this literally from scratch with an architecture we built from scratch, I didn't look out this architecture anywhere. It's just the first thing that came to mind. But in terms of where that puts us, 0.45% error is around about the state of the art for this data set as of three or four years ago.

![](lesson7/11.png)

Today MNIST considered a trivially easy dataset, so I'm not saying like wow, we've broken some records here. People have got beyond 0.45% error, but what I'm saying is this kind of ResNet is a genuinely extremely useful network still today. This is really all we use in our fast ImageNet training still. And one of the reasons as well is that it's so popular so the vendors of the library spend a lot of time optimizing it, so things tend to work fast. Where else, some more modern style architectures using things like separable or group convolutions tend not to actually train very quickly in practice. 

![](lesson7/12.png)

If you look at the definition of `res_block` in the fast.ai code, you'll see it looks a little bit different to this, and that's because I've created something called a `MergeLayer`. A `MergeLayer` is something which in the forward (just skip dense for a moment), the forward says `x+x.orig`. So you can see there's something ResNet-ish going on here. What is `x.orig`? Well, if you create a special kind of sequential model called a `SequentialEx`  so this is like fast.ai's sequential extended. It's just like a normal sequential model, but we store the input in `x.orig`. So this `SequentialEx`, `conv_layer`, `conv_layer`, `MergeLayer`, will do exactly the same as `ResBlock`. So you can create your own variations of ResNet blocks very easily with this `SequentialEx` and `MergeLayer`.

There's something else here which is when you create your MergeLayer, you can optionally set `dense=True`, and what happens if you do? Well, if you do, it doesn't go `x+x.orig`, it goes `cat([x,x.orig])`. In other words, rather than putting a plus in this connection, it does a concatenate. That's pretty interesting because what happens is that you have your input coming in to your Res block, and once you use concatenate instead of plus, it's not called a Res block anymore, it's called a dense block. And it's not called a ResNet anymore, it's called a DenseNet.

The DenseNet was invented about a year after the ResNet, and if you read the DenseNet paper, it can sound incredibly complex and different, but actually it's literally identical but plus here is placed with cat. So you have your input coming into your dense block, and you've got a few convolutions in here, and then you've got some output coming out, and then you've got your identity connection, and remember it doesn't plus, it concats so the channel axis gets a little bit bigger. Then we do another dense block, and at the end of that, we have the result of the convolution as per usual, but this time the identity block is that big.

![](lesson7/13.png)

So you can see that what happens is that with dense blocks it's getting bigger and bigger and bigger, and kind of interestingly the exact input is still here. So actually, no matter how deep you get the original input pixels are still there, and the original layer 1 features are still there, and the original layer 2 features are still there. So as you can imagine, DenseNets are very memory intensive. There are ways to manage this. From time to time, you can have a regular convolution and it squishes your channels back down, but they are memory intensive. But, they have very few parameters. So for dealing with small datasets, you should definitely experiment with dense blocks and DenseNets. They tend to work really well on small datasets.

Also, because it's possible to keep those original input pixels all the way down the path, they work really well for segmentation. Because for segmentation, you want to be able to reconstruct the original resolution of your picture, so having all of those original pixels still there is a super helpful.

### U-Net [[30:16](https://youtu.be/nWpdkZE2_cc?t=1816)]

That's ResNets. One of the main reasons other than the fact that ResNets are awesome to tell you about them is that these skipped connections are useful in other places as well. They are particularly useful in other places in other ways of designing architectures for segmentation. So in building this lesson, I keep trying to take old papers and imagining like what would that person have done if they had access to all the modern techniques we have now, and I try to rebuild them in a more modern style. So I've been really rebuilding this next architecture we're going to look at called U-Net in a more modern style recently, and got to the point now I keep showing you this semantic segmentation paper with the state of the art for CamVid which was 91.5.

![](lesson7/14.png)

This week, I got it up to 94.1 using the architecture I'm about to show you. So we keep pushing this further and further and further. And it's really it was all about adding all of the modern tricks - many of which I'll show you today, some of which we will see in part 2.

What we're going to do to get there is we're going to use this U-Net. We've used a U-Net before. We used it when we did the CamVid segmentation but we didn't understand what it was doing. So we're now in a position where we can understand what it was doing. The first thing we need to do is to understand the basic idea of how you can do segmentation. If we go back to our [CamVid notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-camvid-tiramisu.ipynb), in our CamVid notebook you'll remember that basically what we were doing is we were taking these photos and adding a class to every single pixel. 

```python
bs,size = 8,src_size//2
```

```python
src = (SegmentationItemList.from_folder(path)
       .split_by_folder(valid='val')
       .label_from_func(get_y_fn, classes=codes))
```

```python
data = (src.transform(get_transforms(), tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))
```

```python
data.show_batch(2, figsize=(10,7))
```

![](lesson7/15.png)

So when you go `data.show_batch` for something which is a `SegmentationItemList`, it will automatically show you these color-coded pixels.

[[32:35](https://youtu.be/nWpdkZE2_cc?t=1955)]

Here's the thing. In order to color code this as a pedestrian, but this as a bicyclist, it needs to know what it is. It needs to actually know that's what a pedestrian looks like, and it needs to know that's exactly where the pedestrian is, and this is the arm of the pedestrian and not part of their shopping basket. It needs to really understand a lot about this picture to do this task, and it really does do this task. When you looked at the results of our top model, I can't see a single pixel by looking at it by eye, I know there's a few wrong, but I can't see the ones that are wrong. It's that accurate. So how does it do that?

The way that we're doing it to get these really really good results is not surprisingly using pre-training. 

```python
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
```

```python
metrics=acc_camvid
wd=1e-2
```

```python
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
```

So we start with a ResNet 34 and you can see that here `unet_learner(data, models.resnet34,...)`. If you don't say `pretrained=False` , by default, you get `pretrained=True` because ... why not?

![](lesson7/16.png)

We start with a ResNet 34 which starts with a big image. In this case, this is from the U-Net paper. Their images, they started with one channel by 572 by 572. This is for medical imaging segmentation. After your stride 2 conv, they're doubling the number of channels to 128, and they're halving the size so they're now down to 280 by 280. In this original unit paper, they didn't add any padding. So they lost a pixel on each side each time they did a conv. That's why you are losing these two. But basically half the size, and then half the size, and then half the size, and then half the size, until they're down to 28 by 28 with 1024 channels.

So that's what the U-Net's downsampling path (the left half is called the downsampling path) look like. Ours is just a ResNet 34. So you can see it here `learn.summary()`, this is literally a ResNet 34. So you can see that the size keeps halving, channels keep going up and so forth.

```python
learn.summary()
```

```
======================================================================
Layer (type)         Output Shape         Param #    Trainable 
======================================================================
Conv2d               [8, 64, 180, 240]    9408       False     
______________________________________________________________________
BatchNorm2d          [8, 64, 180, 240]    128        True      
______________________________________________________________________
ReLU                 [8, 64, 180, 240]    0          False     
______________________________________________________________________
MaxPool2d            [8, 64, 90, 120]     0          False     
______________________________________________________________________
Conv2d               [8, 64, 90, 120]     36864      False     
______________________________________________________________________
BatchNorm2d          [8, 64, 90, 120]     128        True      
______________________________________________________________________
ReLU                 [8, 64, 90, 120]     0          False     
______________________________________________________________________
Conv2d               [8, 64, 90, 120]     36864      False     
______________________________________________________________________
BatchNorm2d          [8, 64, 90, 120]     128        True      
______________________________________________________________________
Conv2d               [8, 64, 90, 120]     36864      False     
______________________________________________________________________
BatchNorm2d          [8, 64, 90, 120]     128        True      
______________________________________________________________________
ReLU                 [8, 64, 90, 120]     0          False     
______________________________________________________________________
Conv2d               [8, 64, 90, 120]     36864      False     
______________________________________________________________________
BatchNorm2d          [8, 64, 90, 120]     128        True      
______________________________________________________________________
Conv2d               [8, 64, 90, 120]     36864      False     
______________________________________________________________________
BatchNorm2d          [8, 64, 90, 120]     128        True      
______________________________________________________________________
ReLU                 [8, 64, 90, 120]     0          False     
______________________________________________________________________
Conv2d               [8, 64, 90, 120]     36864      False     
______________________________________________________________________
BatchNorm2d          [8, 64, 90, 120]     128        True      
______________________________________________________________________
Conv2d               [8, 128, 45, 60]     73728      False     
______________________________________________________________________
BatchNorm2d          [8, 128, 45, 60]     256        True      
______________________________________________________________________
ReLU                 [8, 128, 45, 60]     0          False     
______________________________________________________________________
Conv2d               [8, 128, 45, 60]     147456     False     
______________________________________________________________________
BatchNorm2d          [8, 128, 45, 60]     256        True      
______________________________________________________________________
Conv2d               [8, 128, 45, 60]     8192       False     
______________________________________________________________________
BatchNorm2d          [8, 128, 45, 60]     256        True      
______________________________________________________________________
Conv2d               [8, 128, 45, 60]     147456     False     
______________________________________________________________________
BatchNorm2d          [8, 128, 45, 60]     256        True      
______________________________________________________________________
ReLU                 [8, 128, 45, 60]     0          False     
______________________________________________________________________
Conv2d               [8, 128, 45, 60]     147456     False     
______________________________________________________________________
BatchNorm2d          [8, 128, 45, 60]     256        True      
______________________________________________________________________
Conv2d               [8, 128, 45, 60]     147456     False     
______________________________________________________________________
BatchNorm2d          [8, 128, 45, 60]     256        True      
______________________________________________________________________
ReLU                 [8, 128, 45, 60]     0          False     
______________________________________________________________________
Conv2d               [8, 128, 45, 60]     147456     False     
______________________________________________________________________
BatchNorm2d          [8, 128, 45, 60]     256        True      
______________________________________________________________________
Conv2d               [8, 128, 45, 60]     147456     False     
______________________________________________________________________
BatchNorm2d          [8, 128, 45, 60]     256        True      
______________________________________________________________________
ReLU                 [8, 128, 45, 60]     0          False     
______________________________________________________________________
Conv2d               [8, 128, 45, 60]     147456     False     
______________________________________________________________________
BatchNorm2d          [8, 128, 45, 60]     256        True      
______________________________________________________________________
Conv2d               [8, 256, 23, 30]     294912     False     
______________________________________________________________________
BatchNorm2d          [8, 256, 23, 30]     512        True      
______________________________________________________________________
ReLU                 [8, 256, 23, 30]     0          False     
______________________________________________________________________
Conv2d               [8, 256, 23, 30]     589824     False     
______________________________________________________________________
BatchNorm2d          [8, 256, 23, 30]     512        True      
______________________________________________________________________
Conv2d               [8, 256, 23, 30]     32768      False     
______________________________________________________________________
BatchNorm2d          [8, 256, 23, 30]     512        True      
______________________________________________________________________
Conv2d               [8, 256, 23, 30]     589824     False     
______________________________________________________________________
BatchNorm2d          [8, 256, 23, 30]     512        True      
______________________________________________________________________
ReLU                 [8, 256, 23, 30]     0          False     
______________________________________________________________________
Conv2d               [8, 256, 23, 30]     589824     False     
______________________________________________________________________
BatchNorm2d          [8, 256, 23, 30]     512        True      
______________________________________________________________________
Conv2d               [8, 256, 23, 30]     589824     False     
______________________________________________________________________
BatchNorm2d          [8, 256, 23, 30]     512        True      
______________________________________________________________________
ReLU                 [8, 256, 23, 30]     0          False     
______________________________________________________________________
Conv2d               [8, 256, 23, 30]     589824     False     
______________________________________________________________________
BatchNorm2d          [8, 256, 23, 30]     512        True      
______________________________________________________________________
Conv2d               [8, 256, 23, 30]     589824     False     
______________________________________________________________________
BatchNorm2d          [8, 256, 23, 30]     512        True      
______________________________________________________________________
ReLU                 [8, 256, 23, 30]     0          False     
______________________________________________________________________
Conv2d               [8, 256, 23, 30]     589824     False     
______________________________________________________________________
BatchNorm2d          [8, 256, 23, 30]     512        True      
______________________________________________________________________
Conv2d               [8, 256, 23, 30]     589824     False     
______________________________________________________________________
BatchNorm2d          [8, 256, 23, 30]     512        True      
______________________________________________________________________
ReLU                 [8, 256, 23, 30]     0          False     
______________________________________________________________________
Conv2d               [8, 256, 23, 30]     589824     False     
______________________________________________________________________
BatchNorm2d          [8, 256, 23, 30]     512        True      
______________________________________________________________________
Conv2d               [8, 256, 23, 30]     589824     False     
______________________________________________________________________
BatchNorm2d          [8, 256, 23, 30]     512        True      
______________________________________________________________________
ReLU                 [8, 256, 23, 30]     0          False     
______________________________________________________________________
Conv2d               [8, 256, 23, 30]     589824     False     
______________________________________________________________________
BatchNorm2d          [8, 256, 23, 30]     512        True      
______________________________________________________________________
Conv2d               [8, 512, 12, 15]     1179648    False     
______________________________________________________________________
BatchNorm2d          [8, 512, 12, 15]     1024       True      
______________________________________________________________________
ReLU                 [8, 512, 12, 15]     0          False     
______________________________________________________________________
Conv2d               [8, 512, 12, 15]     2359296    False     
______________________________________________________________________
BatchNorm2d          [8, 512, 12, 15]     1024       True      
______________________________________________________________________
Conv2d               [8, 512, 12, 15]     131072     False     
______________________________________________________________________
BatchNorm2d          [8, 512, 12, 15]     1024       True      
______________________________________________________________________
Conv2d               [8, 512, 12, 15]     2359296    False     
______________________________________________________________________
BatchNorm2d          [8, 512, 12, 15]     1024       True      
______________________________________________________________________
ReLU                 [8, 512, 12, 15]     0          False     
______________________________________________________________________
Conv2d               [8, 512, 12, 15]     2359296    False     
______________________________________________________________________
BatchNorm2d          [8, 512, 12, 15]     1024       True      
______________________________________________________________________
Conv2d               [8, 512, 12, 15]     2359296    False     
______________________________________________________________________
BatchNorm2d          [8, 512, 12, 15]     1024       True      
______________________________________________________________________
ReLU                 [8, 512, 12, 15]     0          False     
______________________________________________________________________
Conv2d               [8, 512, 12, 15]     2359296    False     
______________________________________________________________________
BatchNorm2d          [8, 512, 12, 15]     1024       True      
______________________________________________________________________
BatchNorm2d          [8, 512, 12, 15]     1024       True      
______________________________________________________________________
ReLU                 [8, 512, 12, 15]     0          False     
______________________________________________________________________
Conv2d               [8, 1024, 12, 15]    4719616    True      
______________________________________________________________________
ReLU                 [8, 1024, 12, 15]    0          False     
______________________________________________________________________
Conv2d               [8, 512, 12, 15]     4719104    True      
______________________________________________________________________
ReLU                 [8, 512, 12, 15]     0          False     
______________________________________________________________________
Conv2d               [8, 1024, 12, 15]    525312     True      
______________________________________________________________________
PixelShuffle         [8, 256, 24, 30]     0          False     
______________________________________________________________________
ReplicationPad2d     [8, 256, 25, 31]     0          False     
______________________________________________________________________
AvgPool2d            [8, 256, 24, 30]     0          False     
______________________________________________________________________
ReLU                 [8, 1024, 12, 15]    0          False     
______________________________________________________________________
BatchNorm2d          [8, 256, 23, 30]     512        True      
______________________________________________________________________
Conv2d               [8, 512, 23, 30]     2359808    True      
______________________________________________________________________
ReLU                 [8, 512, 23, 30]     0          False     
______________________________________________________________________
Conv2d               [8, 512, 23, 30]     2359808    True      
______________________________________________________________________
ReLU                 [8, 512, 23, 30]     0          False     
______________________________________________________________________
ReLU                 [8, 512, 23, 30]     0          False     
______________________________________________________________________
Conv2d               [8, 1024, 23, 30]    525312     True      
______________________________________________________________________
PixelShuffle         [8, 256, 46, 60]     0          False     
______________________________________________________________________
ReplicationPad2d     [8, 256, 47, 61]     0          False     
______________________________________________________________________
AvgPool2d            [8, 256, 46, 60]     0          False     
______________________________________________________________________
ReLU                 [8, 1024, 23, 30]    0          False     
______________________________________________________________________
BatchNorm2d          [8, 128, 45, 60]     256        True      
______________________________________________________________________
Conv2d               [8, 384, 45, 60]     1327488    True      
______________________________________________________________________
ReLU                 [8, 384, 45, 60]     0          False     
______________________________________________________________________
Conv2d               [8, 384, 45, 60]     1327488    True      
______________________________________________________________________
ReLU                 [8, 384, 45, 60]     0          False     
______________________________________________________________________
ReLU                 [8, 384, 45, 60]     0          False     
______________________________________________________________________
Conv2d               [8, 768, 45, 60]     295680     True      
______________________________________________________________________
PixelShuffle         [8, 192, 90, 120]    0          False     
______________________________________________________________________
ReplicationPad2d     [8, 192, 91, 121]    0          False     
______________________________________________________________________
AvgPool2d            [8, 192, 90, 120]    0          False     
______________________________________________________________________
ReLU                 [8, 768, 45, 60]     0          False     
______________________________________________________________________
BatchNorm2d          [8, 64, 90, 120]     128        True      
______________________________________________________________________
Conv2d               [8, 256, 90, 120]    590080     True      
______________________________________________________________________
ReLU                 [8, 256, 90, 120]    0          False     
______________________________________________________________________
Conv2d               [8, 256, 90, 120]    590080     True      
______________________________________________________________________
ReLU                 [8, 256, 90, 120]    0          False     
______________________________________________________________________
ReLU                 [8, 256, 90, 120]    0          False     
______________________________________________________________________
Conv2d               [8, 512, 90, 120]    131584     True      
______________________________________________________________________
PixelShuffle         [8, 128, 180, 240]   0          False     
______________________________________________________________________
ReplicationPad2d     [8, 128, 181, 241]   0          False     
______________________________________________________________________
AvgPool2d            [8, 128, 180, 240]   0          False     
______________________________________________________________________
ReLU                 [8, 512, 90, 120]    0          False     
______________________________________________________________________
BatchNorm2d          [8, 64, 180, 240]    128        True      
______________________________________________________________________
Conv2d               [8, 96, 180, 240]    165984     True      
______________________________________________________________________
ReLU                 [8, 96, 180, 240]    0          False     
______________________________________________________________________
Conv2d               [8, 96, 180, 240]    83040      True      
______________________________________________________________________
ReLU                 [8, 96, 180, 240]    0          False     
______________________________________________________________________
ReLU                 [8, 192, 180, 240]   0          False     
______________________________________________________________________
Conv2d               [8, 384, 180, 240]   37248      True      
______________________________________________________________________
PixelShuffle         [8, 96, 360, 480]    0          False     
______________________________________________________________________
ReplicationPad2d     [8, 96, 361, 481]    0          False     
______________________________________________________________________
AvgPool2d            [8, 96, 360, 480]    0          False     
______________________________________________________________________
ReLU                 [8, 384, 180, 240]   0          False     
______________________________________________________________________
MergeLayer           [8, 99, 360, 480]    0          False     
______________________________________________________________________
Conv2d               [8, 49, 360, 480]    43708      True      
______________________________________________________________________
ReLU                 [8, 49, 360, 480]    0          False     
______________________________________________________________________
Conv2d               [8, 99, 360, 480]    43758      True      
______________________________________________________________________
ReLU                 [8, 99, 360, 480]    0          False     
______________________________________________________________________
MergeLayer           [8, 99, 360, 480]    0          False     
______________________________________________________________________
Conv2d               [8, 12, 360, 480]    1200       True      
______________________________________________________________________

Total params:  41133018
Total trainable params:  19865370
Total non-trainable params:  21267648
```

Eventually, you've got down to a point where, if you use U-Net architecture, it's 28 by 28 with 1,024 channels. With the ResNet architecture with a 224 pixel input, it would be 512 channels by 7 by 7. So it's a pretty small grid size on this feature map. Somehow, we've got to end up with something which is the same size as our original picture. So how do we do that? How do you do computation which increases the grid size? Well, we don't have a way to do that in our current bag of tricks. We can use a stride one conv to do computation and keeps grid size or a stride 2 conv to do computation and halve the grid size. 

[[35:58](https://youtu.be/nWpdkZE2_cc?t=2158)]

So how do we double the grid size? We do a **stride half conv**, also known as a **deconvolution**, also known as a **transpose convolution**.

![](lesson7/17.png)

There is a fantastic paper called [A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285.pdf) that shows a great picture of exactly what does a 3x3 kernel stride half conv look like. And it's literally this. If you have a 2x2 input, so the blue squares are the 2x2 input, you add not only 2 pixels of padding all around the outside, but you also add a pixel of padding between every pixel. So now if we put this 3x3 kernel here, and then here, and then here, you see how the 3x3 kernels just moving across it in the usual way, you will end up going from a 2x2 output to a 5x5 output. If you only added one pixel of padding around the outside, you would end up with a 4x4 output. 

This is how you can increase the resolution. This was the way people did it until maybe a year or two ago. There's another trick for improving things you find online. Because this is actually a dumb way to do it. And it's kind of obvious it's a dumb way to do it for a couple of reasons. One is that, have a look at the shaded area on the left, nearly all of those pixels are white. They're nearly all zeros. What a waste. What a waste of time, what a waste of computation. There's just nothing going on there. 

![](lesson7/18.png)

Also, this one when you get down to that 3x3 area, 2 out of the 9 pixels are non-white, but this left one, 1 out of the 9 are non-white. So there's different amounts of information going into different parts of your convolution. So it just doesn't make any sense to throw away information like this and to do all this unnecessary computation and have different parts of the convolution having access to different amounts of information.

What people generally do nowadays is something really simple. If you have a, let's say, 2x2 input with these are your pixel values (a, b, c, d) and you want to create a 4x4, why not just do this?

![](lesson7/19.png)

So I've now up scaled from 2 by 2 to 4 by 4. I haven't done any interesting computation, but now on top of that, I could just do a stride 1 convolution, and now I have done some computation. 

An upsample, this is called **nearest neighbor interpolation**. That's super fast which is nice. So you can do a nearest neighbor interpolation, and then a stride 1 conv, and now you've got some computation which is actually using there's no zeros in upper left 4x4, this (one pixel to the right) is kind of nice because it gets a mixture of A's and B's which is kind of what you would want and so forth.

Another approach is instead of using nearest neighbor interpolation, you can use bilinear interpolation which basically means instead of copying A to all those different cells you take a  weighted average of the cells around it. 

![](lesson7/20.png)

For example if you were looking at what should go here (red), you would kind of go, oh it's about 3 A's, 2 C's, 1 D, and 2 B's, and you take the average, not exactly, but roughly just a weighted average. Bilinear interpolation, you'll find all over the place - it's pretty standard technique. Anytime you look at a picture on your computer screen and change its size, it's doing bilinear interpolation. So you can do that and then a stride 1 conv. So that was what people were using, well, what people still tend to use. That's as much as I going to teach you this part. In part 2, we will actually learn what the fast.ai library is actually doing behind the scenes which is something called a **pixel shuffle** also known as **sub pixel convolutions**. It's not dramatically more complex but complex enough that I won't cover it today. They're the same basic idea. All of these things is something which is basically letting us do a convolution that ends up with something that's twice the size.

That gives us our upsampling path. That lets us go from 28 by 28 to 54 by 54 and keep on doubling the size, so that's good. And that was it until U-Net came along. That's what people did and it didn't work real well which is not surprising because like in this 28 by 28 feature map, how the heck is it going to have enough information to reconstruct a 572 by 572 output space? That's a really tough ask. So you tended to end up with these things that lack fine detail.

[[41:45](https://youtu.be/nWpdkZE2_cc?t=2505)]

![](lesson7/21.png)

So what Olaf Ronneberger et al. did was they said hey let's add a skip connection, an identity connection, and amazingly enough, this was before ResNets existed. So this was like a really big leap, really impressive. But rather than adding a skip connection that skipped every two convolutions, they added skip connections where these gray lines are. In other words, they added a skip connection from the same part of the downsampling path to the same-sized bit in the upsampling path. And they didn't add, that's why you can see the white and the blue next to each other, they didn't add they concatenated. So basically, these are like dense blocks, but the skip connections are skipping over larger and larger amounts of the architecture so that over here (top gray arrow), you've nearly got the input pixels themselves coming into the computation of these last couple of layers. That's going to make it super handy for resolving the fine details in these segmentation tasks because you've literally got all of the fine details. On the downside, you don't have very many layers of computation going on here (top right), just four. So you better hope that by that stage, you've done all the computation necessary to figure out is this a bicyclist or is this a pedestrian, but you can then add on top of that something saying is this exact pixel where their nose finishes or is at the start of the tree. So that works out really well and that's U-Net.

[[43:33](https://youtu.be/nWpdkZE2_cc?t=2613)]

![](lesson7/22.png)

This is the unit code from fast.ai, and the key thing that comes in is the encoder. The encoder refers to the downsampling part of U-Net, in other words, in our case a ResNet 34. In most cases they have this specific older style architecture, but like I said, replace any older style architecture bits with ResNet bits and life improves particularly if they're pre-trained. So that certainly happened for us. So we start with our encoder.

So our `layers` of our U-Net is an encoder, then batch norm, then ReLU, and then `middle_conv` which is just (`conv_layer`, `conv_layer`). Remember, `conv_layer` is a conv, ReLU, batch norm in fast.ai. So that middle con is these two extra steps here at the bottom:

![](lesson7/23.png)

It's doing a little bit of computation. It's kind of nice to add more layers of computation where you can. So encoder, batch norm, ReLU, and then two convolutions. Then we enumerate through these indexes (`sfs_idxs`). What are these indexes? I haven't included the code but these are basically we figure out what is the layer number where each of these stride 2 convs occurs and we just store it in an array of indexes. Then we can loop through that and we can basically say for each one of those points create a `UnetBlock` telling us how many upsampling channels that are and how many cross connection. These gray arrows are called cross connections - at least that's what I call them. 

[[45:16](https://youtu.be/nWpdkZE2_cc?t=2716)]

That's really the main works going on in the in the `UnetBlock`. As I said, there's quite a few tweaks we do as well as the fact we use a much better encoder, we also use some tweaks in all of our upsampling using this pixel shuffle, we use another tweak called ICNR, and then another tweak which I just did in the last week is to not just take the result of the convolutions and pass it across, but we actually grab the input pixels and make them another cross connection. That's what this `last_cross` is here. You can see we're literally appending a `res_block` with the original inputs (so you can see our `MergeLayer`). 

![](lesson7/24.png)

So really all the work is going on in a `UnetBlock`  and `UnetBlock` has to store the the activations at each of these downsampling points, and the way to do that, as we learn in the last lesson, is with hooks. So we put hooks into the ResNet 34 to store the activations each time there's a stride 2 conv, and so you can see here, we grab the hook (`self.hook =hook`). And we grab the result of the stored value in that hook, and we literally just go `torch.cat` so we concatenate the upsampled convolution with the result of the hook which we chuck through batch norm, and then we do two convolutions to it.

Actually, something you could play with at home is pretty obvious here (the very last line). Anytime you see two convolutions like this, there's an obvious question is what if we used a ResNet block instead? So you could try replacing those two convs with a ResNet block, you might find you get even better results. They're the kind of things I look for when I look at an architecture is like "oh, two convs in a row, probably should be a ResNet block. 

Okay, so that's U-Net and it's amazing to think it preceded ResNet, preceded DenseNet. It wasn't even published in a major machine learning venue. It was actually published in MICCAI which is a specialized medical image computing conference. For years, it was largely unknown outside of the medical imaging community. Actually, what happened was Kaggle competitions for segmentation kept on being easily won by people using U-Nets and that was the first time I saw it getting noticed outside the medical imaging community. Then gradually, a few people in the academic machine learning community started noticing, and now everybody loves U-Net, which I'm glad because it's just awesome.

So identity connections, regardless of whether they're a plus style or a concat style, are incredibly useful. They can basically get us close to the state of the art on lots of important tasks. So I want to use them on another task now.

### Image restoration [[48:31](https://youtu.be/nWpdkZE2_cc?t=2911)]

The next task I want to look at is image restoration. Image restoration refers to starting with an image and this time we're not going to create a segmentation mask but we're going to try and create a better image. There's lots of kind of versions of better - there could be different image. The kind of things we can do with this kind of image generation would be:

- take a low res image make it high res 
- take a black-and-white image make a color 
- take an image where something's being cut out of it and trying to replace the cutout thing 
- take a photo and try and turn it into what looks like a line drawing 
- take a photo and try and talk like it look like a Monet painting 

These are all examples of image to image generation tasks which you'll know how to do after this part class. 

So in our case, we're going to try to do image restoration which is going to start with low resolution, poor quality JPEGs, with writing written over the top of them, and get them to replace them with high resolution, good quality pictures in which the text has been removed.

**Question**: Why do you concat before calling conv2(conv1(x)), not after? [[49:50](https://youtu.be/nWpdkZE2_cc?t=2990)] 

Because if you did your convs before you concat, then there's no way for the channels of the two parts to interact with each other. So remember in a 2D conv, it's really 3D. It's moving across 2 dimensions but in each case it's doing a dot product of all 3 dimensions of a rank 3 tensor (row by column by channel). So generally speaking, we want as much interaction as possible. We want to say this part of the downsampling path and this part of the upsampling path, if you look at the combination of them, you find these interesting things. So generally you want to have as many interactions going on as possible in each computation that you do.

**Question**: How does concatenating every layer together in a DenseNet work when the size of the image/feature maps is changing through the layers? [[50:54](https://youtu.be/nWpdkZE2_cc?t=3054)]

That's a great question. If you have a stride 2 conv, you can't keep DenseNet-ing. That's what actually happens in a DenseNet is you kind of go like dense block, growing, dense block, growing, dense block, growing, so you are getting more and more channels. Then you do a stride 2 conv without a dense block, and so now it's kind of gone. Then you just do a few more dense blocks and then it's gone. So in practice, a dense block doesn't actually keep all the information all the way through, but just up until every one of these stride 2 convs. There's various ways of doing these bottlenecking layers where you're basically saying hey let's reset. It also helps us keep memory under control because at that point we can decide how many channels we actually want.

#### Back to image restoration [[52:01](https://youtu.be/nWpdkZE2_cc?t=3121)]

[lesson7-superres-gan.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres-gan.ipynb)

In order to create something which can turn crappy images into nice images, we needed dataset containing nice versions of images and crappy versions of the same images. The easiest way to do that is to start with some nice images and "crappify" them. 

```python
from PIL import Image, ImageDraw, ImageFont
```

```python
def crappify(fn,i):
    dest = path_lr/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, 96, use_min=True)
    img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
    w,h = img.size
    q = random.randint(10,70)
    ImageDraw.Draw(img).text((random.randint(0,w//2),random.randint(0,h//2)), str(q), fill=(255,255,255))
    img.save(dest, quality=q)
```

The way to crappify them is to create a function called crappify which contains your crappification logic. My crappification logic, you can pick your own, is that:

- I open up my nice image
- I resize it to be really small 96 by 96 pixels with bilinear interpolation
- I then pick a random number between 10 and 70
- I draw that number into my image at some random location
- Then I save that image with a JPEG quality of that random number. 

A JPEG quality of 10 is like absolute rubbish, a JPEG a quality of 70 is not bad at all. So I end up with high quality images and low quality images that look something like these:

![](lesson7/25.png)

You can see this one (bottom row) there's the number, and this is after transformation so that's why it's been flipped and you won't always see the number because we're zooming into them, so a lot of the time, the number is cropped out. 

It's trying to figure out how to take this incredibly JPEG artifactory thing with text written over the top, and turn it into this (image on the right). I'm using the Oxford pets data set again. The same one we used in lesson one. So there's nothing more high quality than pictures of dogs and cats, I think we can all agree with that. 

#### `parallel` [[53:48](https://youtu.be/nWpdkZE2_cc?t=3228)]

The crappification process can take a while, but fast.ai has a function called `parallel`. If you pass `parallel` a function name and a list of things to run that function on, it will run that function on them all in parallel. So this actually can run pretty quickly.

```python
il = ImageItemList.from_folder(path_hr)
parallel(crappify, il.items)
```

The way you write this `crappify` function is where you get to do all the interesting stuff in this assignment. Try and think of an interesting crappification which does something that you want to do. So if you want to colorize black-and-white images, you would replace it with black-and-white. If you want something which can take large cutout blocks of image and replace them with kind of hallucinatin image, add a big black box to these. If you want something which can take old families photos scans that have been like folded up and have crinkles in, try and find a way of adding dust prints and crinkles and so forth. 

Anything that you don't include in `crappify`, your model won't learn to fix. Because every time it sees that in your photos, the input and output will be the same. So it won't consider that to be something worthy of fixing.

[[55:09](https://youtu.be/nWpdkZE2_cc?t=3309)]

![](lesson7/26.png)

We now want to create a model which can take an input photo that looks like that (left) and output something that looks like that (right). So obviously, what we want to do is use U-Net because we already know that U-Net can do exactly that kind of thing, and we just need to pass the U-Net that data.

```python
arch = models.resnet34
src = ImageImageList.from_folder(path_lr).random_split_by_pct(0.1, seed=42)
```

```python
def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr/x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data
```

```python
data_gen = get_data(bs,size)
```

Our data is just literally the file names from each of those two folders, do some transforms, data bunch, normalize. We'll use ImageNet stats because we're going to use a pre-trained model. Why are we using a pre-trained model? Because if you're going to get rid of this 46, you need to know what probably was there, and to know what probably was there you need to know what this is a picture of. Otherwise, how can you possibly know what it ought to look like. So let's use a pre-trained model that knows about these kinds of things.

```python
wd = 1e-3
```

```python
y_range = (-3.,3.)
```

```python
loss_gen = MSELossFlat()
```

```python
def create_gen_learner():
    return unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen)
```

```python
learn_gen = create_gen_learner()
```

So we created our U-Net with that data, the architecture is ResNet 34. These three things (`blur`, `norm_type`, `self_attention`) are important and interesting and useful, but I'm going to leave them to part 2. For now, you should always include them when you use U-Net for this kind of problem. 

This whole thing, I'm calling a "generator". It's going to generate. This is generative modeling. There's not a really formal definition, but it's basically something where the thing we're outputting is like a real object, in this case an image - it's not just a number. So we're going to create a generator learner which is this U-Net learner, and then we can fit. We're using MSE loss, so in other words what's the mean squared error between the actual pixel value that it should be in the pixel value that we predicted. MSE loss normally expects two vectors. In our case, we have two images so we have a version called MSE loss flat which simply flattens out those images into a big long vector. There's never any reason not to use this, even if you do have a vector, it works fine, if you don't have a vector, it'll also work fine.

```python
learn_gen.fit_one_cycle(2, pct_start=0.8)
```

Total time: 01:35

| epoch | train_loss | valid_loss |
| ----- | ---------- | ---------- |
| 1     | 0.061653   | 0.053493   |
| 2     | 0.051248   | 0.047272   |

We're already down to 0.05 mean squared error on the pixel values which is not bad after 1 minute 35. Like all things in fast.ai pretty much, because we're doing transfer learning by default when you create this, it'll freeze the the pre-trained part. And the pre-trained part of a U-Net is the downsampling part. That's where the ResNet is. 

```python
learn_gen.unfreeze()
```

```python
learn_gen.fit_one_cycle(3, slice(1e-6,1e-3))
```

Total time: 02:24

| epoch | train_loss | valid_loss |
| ----- | ---------- | ---------- |
| 1     | 0.050429   | 0.046088   |
| 2     | 0.049056   | 0.043954   |
| 3     | 0.045437   | 0.043146   |

```python
learn_gen.show_results(rows=4)
```

![](lesson7/27.png)

Let's unfreeze that and train a little more and look at that! With four minutes of training, we've got something which is basically doing a perfect job of removing numbers. It's certainly not doing a good job of upsampling, but it's definitely doing a nice job. Sometimes when it removes a number, it maybe leaves a little bit of JPEG artifact. But it's certainly doing something pretty useful. So if all we wanted to do was kind of watermark removal, we would be finished.

We're not finished because we actually want this thing (middle) to look more like this thing (right). So how are we going to do that? The reason that we're not making as much progress with that as we'd like is that our loss function doesn't really describe what we want. Because actually, the mean squared error between the pixels of this (middle) and this (right) is actually very small. If you actually think about it, most of the pixels are very nearly the right color. But we're missing the texture of the pillow, and we're missing the eyeballs entirely pretty much. We're missing the texture of the fur. So we want some lost function that does a better job than pixel mean squared error loss of saying like is this a good quality picture of this thing.

### Generative Adversarial Network [[59:23](https://youtu.be/nWpdkZE2_cc?t=3563)]

There's a fairly general way of answering that question, and it's something called a Generative adversarial network or GAN. A GAN tries to solve this problem by using a loss function which actually calls another model. Let me describe it to you.

![](lesson7/28.png)

We've got our crappy image, and we've already created a generator. It's not a great one, but it's not terrible and that's creating predictions (like the middle picture). We have a high-res image (like the right picture) and we can compare the high-res image to the prediction with pixel MSE.

We could also train another model which we would variously call either the discriminator or the critic - they both mean the same thing. I'll call it a critic. We could try and build a binary classification model that takes all the pairs of the generated image and the real high-res image, and learn to classify which is which. So look at some picture and say "hey, what do you think? Is that a high-res cat or is that a generated cat? How about this one? Is that a high-res cat or a generated cat?" So just a regular standard binary cross-entropy classifier. We know how to do that already. If we had one of those, we could fine tune the generator and rather than using pixel MSE is the loss, the loss could be <u>how good are we at fooling the critic?</u> Can we create generated images that the critic thinks are real? 

That would be a very good plan, because if it can do that, if the loss function is "am I fooling the critic?"  then it's going to learn to create images which the critic can't tell whether they're real or fake. So we could do that for a while, train a few batches. But the critic isn't that great. The reason the critic isn't that great is because it wasn't that hard. These images are really crappy, so it's really easy to tell the difference. So after we train the generator a little bit more using that critic as the loss function, the generators going to get really good at falling the critic. So now we're going to stop training the generator, and we'll train the critic some more on these newly generated images. Now that the generator is better, it's now a tougher task for the critic to decide which is real and which is fake. So we'll train that a little bit more. Then once we've done that and the critic is now pretty good at recognizing the difference between the better generated images and the originals, we'll go back and we'll fine tune the generator some more using the better discriminator (i.e. the better critic) as the loss function.

So we'll just go ping pong ping pong, backwards and forwards. That's a GAN. That's our version of GAN. I don't know if anybody's written this before, we've created a new version of GAN which is kind of a lot like the original GANs but we have this neat trick where we pre-train the generator and we pre-train the critic. GANs have been kind of in the news a lot. They're pretty fashionable tool, and if you've seen them, you may have heard that they're a real pain to train. But it turns out we realized that really most of the pain of training them was at the start. If you don't have a pre-trained generator and you don't have a pre-trained critic, then it's basically the blind leading the blind. The generator is trying to generate something which fools a critic, but the critic doesn't know anything at all, so it's basically got nothing to do. Then the critic is trying to decide whether the generated images are real or not, and that's really obvious so that just does it. So they don't go anywhere for ages. Then once they finally start picking up steam, they go along pretty quickly,

If you can find a way to generate things without using a GAN like mean squared pixel loss, and discriminate things without using a GAN like predict on that first generator, you can make a lot of progress.

#### Creating a critic/discriminator [[1:04:04](https://youtu.be/nWpdkZE2_cc?t=3844)]

Let's create the critic. To create just a totally standard fast.ai binary classification model, we need two folders; one folder containing high-res images, one folder containing generated images. We already have the folder with high-res images, so we just have to save our generated images.

```python
name_gen = 'image_gen'
path_gen = path/name_gen
```

```python
# shutil.rmtree(path_gen)
```

```python
path_gen.mkdir(exist_ok=True)
```

```python
def save_preds(dl):
    i=0
    names = dl.dataset.items
    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen/names[i].name)
            i += 1
```

```python
save_preds(data_gen.fix_dl)
```

Here's a teeny tiny bit of code that does that. We're going to create a directory called `image_gen`, pop it into a variable called `path_gen`. We got a little function called `save_preds` which takes a data loader. We're going to grab all of the file names. Because remember that in an item list, `.items` contains the file names if it's an image item list. So here's the file names in that data loader's dataset. Now let's go through each batch of the data loader, and let's grab a batch of predictions for that batch, and then `reconstruct=True` means it's actually going to create fast.ai image objects for each thing in the batch. Then we'll go through each of those predictions and save them. The name we'll save it with is the name of the original file, but we're going to pop it into our new directory.

That's it. That's how you save predictions. So you can see, I'm increasingly not just using stuff that's already in the fast.ai library, but try to show you how to write stuff yourself. And generally it doesn't require heaps of code to do that. So if you come back for part 2, lots of part 2 are like here's how you use things inside the library, and of course, here's how we wrote the library. So increasingly, writing our own code.

Okay, so save those predictions and let's just do a `PIL.Image.open` on the first one, and yep there it is. So there's an example of the generated image.

```python
PIL.Image.open(path_gen.ls()[0])
```

![](lesson7/29.png)



Now I can train a critic in the usual way. It's really annoying to have to restart Jupyter notebook to reclaim GPU memory. One easy way to handle this is if you just set something that you knew was using a lot of GPU to `None` like this learner, and then just go `gc.collect`, that tells Python to do memory garbage collection, and after that you'll generally be fine. You'll be able to use all of your GPU memory again.

```python
learn_gen=None
gc.collect()
```

If you're using `nvidia-smi` to actually look at your GPU memory, you won't see it clear because PyTorch still has a kind of allocated cache, but it makes it available. So you should find this is how you can avoid restarting your notebook.

```python
def get_crit_data(classes, bs, size):
    src = ImageItemList.from_folder(path, include=classes).random_split_by_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(max_zoom=2.), size=size)
           .databunch(bs=bs).normalize(imagenet_stats))
    data.c = 3
    return data
```

We're going to create our critic. It's just an image item list from folder in the totally usual way, and the classes will be the `image_gen` and `images`. We will do a random split because we want to know how well we're doing with the critic to have a validation set. We just label it from folder in the usual way, add some transforms, data bunch, normalized. So we've got a totally standard classifier. Here's what some of it looks like:

```python
data_crit = get_crit_data([name_gen, 'images'], bs=bs, size=size)
```

```python
data_crit.show_batch(rows=3, ds_type=DatasetType.Train, imgsize=3)
```

![](lesson7/30.png)

Here's one from the real images, real images, generated images, generated images, ... So it's got to try and figure out which class is which.

```python
loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())
```

We're going to use binary cross entropy as usual. However, we're not going to use a ResNet here. The reason, we'll get into in more detail in part 2, but basically when you're doing a GAN, you need to be particularly careful that the generator and the critic can't both push in the same direction and increase the weights out of control. So we have to use something called spectral normalization to make GANs work nowadays. We'll learned about that in part 2.

```python
def create_critic_learner(data, metrics):
    return Learner(data, gan_critic(), metrics=metrics, loss_func=loss_critic, wd=wd)
```

```python
learn_critic = create_critic_learner(data_crit, accuracy_thresh_expand)
```

Anyway, if you say `gan_critic`, fast.ai will give you a binary classifier suitable for GANs. I strongly suspect we probably can use a ResNet here. We just have to create a pre trained ResNet with spectral norm. Hope to do that pretty soon. We'll see how we go, but as of now, this is kind of the best approach is this thing called `gan_critic`. A GAN critic uses a slightly different way of averaging the different parts of the image when it does the loss, so anytime you're doing a GAN at the moment, you have to wrap your loss function with `AdaptiveLoss`. Again, we'll look at the details in part 2. For now, just know this is what you have to do and it'll work.

Other than that slightly odd loss function and that slightly odd architecture, everything else is the same. We can call that (`create_critic_learner` function) to create our critic. Because we have this slightly different architecture and slightly different loss function, we did a slightly different metric. This is the equivalent GAN version of accuracy for critics. Then we can train it and you can see it's 98% accurate at recognizing that kind of crappy thing from that kind of nice thing. But of course we don't see the numbers here anymore. Because these are the generated images, the generator already knows how to get rid of those numbers that are written on top. 

```python
learn_critic.fit_one_cycle(6, 1e-3)
```

Total time: 09:40

| epoch | train_loss | valid_loss | accuracy_thresh_expand |
| ----- | ---------- | ---------- | ---------------------- |
| 1     | 0.678256   | 0.687312   | 0.531083               |
| 2     | 0.434768   | 0.366180   | 0.851823               |
| 3     | 0.186435   | 0.128874   | 0.955214               |
| 4     | 0.120681   | 0.072901   | 0.980228               |
| 5     | 0.099568   | 0.107304   | 0.962564               |
| 6     | 0.071958   | 0.078094   | 0.976239               |

#### Finishing up GAN [[1:09:52](https://youtu.be/nWpdkZE2_cc?t=4192)]

```python
learn_crit=None
learn_gen=None
gc.collect()
```

```python
data_crit = get_crit_data(['crappy', 'images'], bs=bs, size=size)
```

```python
learn_crit = create_critic_learner(data_crit, metrics=None).load('critic-pre2')
```

```python
learn_gen = create_gen_learner().load('gen-pre2')
```

Let's finish up this GAN. Now that we have pre-trained the generator and pre-trained the critic, we now need to get it to kind of ping pong between training a little bit of each. The amount of time you spend on each of those things and the learning rates you use is still a little bit on the fuzzy side, so we've created a `GANLearner` for you which you just pass in your generator and your critic (which we've just simply loaded here from the ones we just trained) and it will go ahead and when you go `learn.fit`, it will do that for you - it'll figure out how much time to train generator and then when to switch to training the discriminator/critic and it'll go backward and forward.

```python
switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1.,50.), show_img=False, switcher=switcher,
                                 opt_func=partial(optim.Adam, betas=(0.,0.99)), wd=wd)
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))
```

[[1:10:43](https://youtu.be/nWpdkZE2_cc?t=4243)]

 These weights here (`weights_gen=(1.,50.)`) is that, what we actually do is we don't only use the critic as the loss function. If we only use the critic as the loss function, the GAN could get very good at creating pictures that look like real pictures, but they actually have nothing to do with the original photo at all. So we actually add together the pixel loss and the critic loss. Those two losses on different scales, so we multiplied the pixel loss by something between about 50 and about 200 - something in that range generally works pretty well.

Something else with GANs. **GANs hate momentum** when you're training them. It kind of doesn't make sense to train them with momentum because you keep switching between generator and critic, so it's kind of tough. Maybe there are ways to use momentum, but I'm not sure anybody's figured it out. So this number here (`betas=(0.,...)`) when you create an Adam optimizer is where the momentum goes, so you should set that to zero. 

Anyways, if you're doing GANs, use these hyper parameters. It should work okay.