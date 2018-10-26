# Lesson 1
[Webpage](http://course-v3.fast.ai/) / [Video](https://youtu.be/BWWm4AzsdLk)



## Welcome! 

Make sure your GPU environment is set up and you can run Jupyter Notebook

[00_notebook_tutorial.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/00_notebook_tutorial.ipynb)

 
Four shortcuts:

- <kbd>Shift</kbd>+<kbd>Enter</kbd>: Runs the code or markdown on a cell

- <kbd>Up Arrow</kbd>+<kbd>Down Arrow</kbd>: Toggle across cells

- <kbd>b</kbd>: Create new cell

- <kbd>0</kbd>+<kbd>0</kbd>: Restart Kernel


[[2:45](https://youtu.be/BWWm4AzsdLk?t=165)] 

Jupyter Notebook is a really interesting device for data scientists because it lets you run interactive experiments and give you not just a static piece of information but something you can interactively experiment with.

How to use notebooks and the materials well based on the last three years of experience:

1. Just watch a lesson end to end. 
   - Don't try to follow along because it's not really designed to go the speed where you can follow along. It's designed to be something where you just take in the information, you get a general sense of all the pieces, how it all fits together.
   - Then you can go back and go through it more slowly pausing the video, trying things out, making sure that you can do the things that I'm doing and you can try and extend them to do things in your own way.
   - Don't try and stop and understand everything the first time. 

[[4:31](https://youtu.be/BWWm4AzsdLk?t=271)]

You can do world-class practitioner level deep learning. 

![](lesson1/1.png)

Main places to be looking for things are:
- [http://course-v3.fast.ai/](http://course-v3.fast.ai/)
- [https://forums.fast.ai/](https://forums.fast.ai/latest)



[[5:27](https://youtu.be/BWWm4AzsdLk?t=327)]

A little bit about why we should listen to Jeremy:

![](lesson1/2.png)



[[6:48](https://youtu.be/BWWm4AzsdLk?t=408)]

Using machine learning to do useful things:

![](lesson1/3.png)



[[7:26](https://youtu.be/BWWm4AzsdLk?t=446)]

![](lesson1/4.png)

If you follow along with 10 hours a week or so approach for the 7 weeks, by the end, you will be able to:
 
1. Build an image classification model on pictures that you choose that will work at a world class level
2. Classify text using whatever datasets you're interested in
3. Make predictions of commercial applications like sales
4. Build recommendation systems such as the one used by Netflix

Not toy examples of any of these but actually things that can come top 10 in Kaggle competitions, that can beat everything that's in the academic community. 

The prerequisite is one year of coding and high school math.




[[9:05](https://youtu.be/BWWm4AzsdLk?t=545)]

What people say about deep learning which are either pointless or untrue:

![](lesson1/5.png)

- It's not a black box. It's really great for interpreting what's going on.
- It does not need much data for most practical applications.
- You don't need a PhD. Rachel has one so it doesn't actually stop you from doing deep learning if you have a PhD.
- It can be used very widely for lots of different applications, not just for vision.
- You don't need lots of hardware. 36 cents an hour server is more than enough to get world-class results for most problems.
- It is true that maybe this is not going to help you build a sentient brain, but that's not our focus. We are focused on solving interesting real-world problems.



[[10:24](https://youtu.be/BWWm4AzsdLk?t=624)]

![](lesson1/6.png)

Baseball vs. Cricket - An example by Nikhil of what you are going to be able to do by the end of lesson 1:


[[11:02](https://youtu.be/BWWm4AzsdLk?t=662)]

![](lesson1/7.png)

We are going to start by looking at code which is different to many of academic courses. We are going to learn to build a useful thing today. That means that at the end of today, you won't know all the theory. There will be lots of aspects of what we do that you don't know why or how it works. That's okay! You will learn why and how it works over the next 7 weeks. But for now, we've found that what works really well is to actually get your hands dirty coding - not focusing on theory. 

## What's your pet [[12:26](https://youtu.be/BWWm4AzsdLk?t=746)]

[lesson1-pets.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb)

<kbd>Shift</kbd>+<kbd>Enter</kbd> to run a cell

These three lines is what we start every notebook with:
```
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```
These things starting `%` are special directives to Jupyter Notebook itself, they are not Python code. They are called "magics."

- If somebody changes underlying library code while I'm running this, please reload it automatically
- If somebody asks to plot something, then please plot it here in this Jupyter Notebook

The next two lines load up the fastai library:

```python
from fastai import *
from fastai.vision import *
```

What is fastai library? [http://docs.fast.ai/](http://docs.fast.ai/)

Everything we are going to do is going to be using either fastai or [PyTorch](https://pytorch.org/) which fastai sits on top of. PyTorch is fast growing extremely popular library. We use it because we used to use TensorFlow a couple years ago and we found we can do a lot more, a lot more quickly with PyTorch. 

Currently fastai supports four applications:

1. Computer vision
2. Natural language text
3. Tabular data
4. Collaborative filtering


[[15:45](https://youtu.be/BWWm4AzsdLk?t=945)]

`import *` - something you've all been told to never ever do.

There are very good reasons to not use `import *` in standard production code with most libraries. But things like MATLAB is the opposite. Everything is there for you all the time. You don't even have to import things a lot of the time. It's kind of funny - we've got these two extremes of how do I code. The scientific programming community has one way, and then software engineering community has the other. Both have really good reasons for doing things. 

With the fastai library, we actually support both approaches. In Jupyter Notebook where you want to be able to quickly interactively try stuff out, you don't want to constantly going back up to the top and importing more stuff. You want to be able to use lots of tab complete and be very experimental, so `import *` is great. When you are building stuff in production, you can do the normal PEP8 style proper software engineering practices. This is a different style of coding. It's not that there are no rules in data science programming, the rules are different. When you're training models, the most important thing is to be able to interactively experiment quickly. So you will see we use a lot of different processes, styles, and stuff to what you are used to. But they are there for a reason and you'll learn about them over time. 

The other thing to mention is that the fastai library is designed in a very interesting modular way and when you do use import *, there's far less clobbering of things than you might expect. It's all explicitly designed to allow you to pull in things and use them quickly without having problems.

## Looking at the data [[17:56](https://youtu.be/BWWm4AzsdLk?t=1076)]

Two main places that we will be tending to get data from for the course:

1. Academic datasets
    - Academic datasets are really important. They are really interesting. They are things where academics spend a lot of time curating and gathering a dataset so that they can show how well different kinds of approaches work with that data. The idea is they try to design datasets that are challenging in some way and require some kind of breakthrough to do them well. 
    - We are going to start with an academic dataset called the pet dataset.
2. Kaggle competition datasets
  
Both types of datasets are interesting for us particularly because they provide strong baseline. That is to say you want to know if you are doing a good job. So with Kaggle datasets that come from a competition, you can actually submit your results to Kaggle and see how well you would have gone in that competition. If you can get in about the top 10%, then I'd say you are doing pretty well.

Academic datasets, academics write down in papers what the state of the art is so how well did they go with using models on that dataset. So this is what we are going to do. We are going to try to create models that get right up towards the top of Kaggle competitions, preferably in the top 10, not just top 10% or that meet or exceed academic state-of-the-art published results. So when you use an academic dataset, it's important to cite it. You don't need to read that paper right now, but if you are interested in learning more about it and why it was created and how it was created, all the details are there. 

Pet dataset is going to ask us to distinguish between 37 different categories of dog breed and cat breed. So that's really hard. In fact, every course until this one, we've used a different dataset which is one where you just have to decide if something is a dog or a cat. So you've got a 50-50 chance right away and dogs and cats look really different. Or else lots of dog breeds and cat breeds look pretty much the same. 

So why have we changed the dataset? We've got to the point now where deep learning os so fast and so easy that the dogs versus cats problem which a few years ago was considered extremely difficult ~80% accuracy was the state of the art, it's now too easy. Our models were basically getting everything right all the time without any tuning and so there weren't really a lot of opportunities for me to show you how to do more sophisticated stuff. So we've picked a harder problem this year. 



[[20:51](https://youtu.be/BWWm4AzsdLk?t=1251)]

This kind of thing where you have to distinguish between similar categories is called fine grained classification in the academic context.  

### untar_data

The first thing we have to do is download and extract the data that we want. We're going to be using this function called `untar_data` which will download it automatically and untar it. AWS has been kind enough to give us lots of space and bandwidth for these datasets so they'll download super quickly for you.

```python
path = untar_data(URLs.PETS); path
```

### help 

The first question then would be how do I know what `untar_data` does. You could just type help and you will find out what module it came from (since we did `import *` you don't necessarily know that), what it does, and something you might not have seen before even if you are an experienced programmer is what exactly you pass to it. You're probably used to seeing the names: url, fname, dest, but you might not be used to seeing `Union[pathlib.Path, str]`. These bits are types and if you're used to typed programming language, you would be used to seeing them, but Python programmers are less used to it. But if you think about it, you don't actually know how to use a function unless you know what type each thing is that you're providing it. So we make sure that we give you that type information directly here in the help. 

In this case, `url` is a string, `fname` is either path or a string and defaults to nothing (`Union` means "either"). `dest` is either a string or a path and defaults to nothing. 

```python
help(untar_data)
```

```
Help on function untar_data in module fastai.datasets:

untar_data(url:str, fname:Union[pathlib.Path, str]=None, dest:Union[pathlib.Path, str]=None)
    Download `url` if doesn't exist to `fname` and un-tgz to folder `dest`
```

We'll learn more shortly about how to get more documentation about the details of this, but for now, we can see we don't have to pass in a file name `fname` or a destination `dest`, it'll figure them out for us from the URL. 

For all the datasets we'll be using in the course, we already have constants defined for all of them. So in this [URLs](https://github.com/fastai/fastai/blob/master/fastai/datasets.py) class, you can see where it's going to grab it from.

`untar_data` will download that to some convenient path and untar it for us and it will then return the value of path. 

```python
path = untar_data(URLs.PETS); path
```
```
PosixPath('/data1/jhoward/git/course-v3/nbs/dl1/data/oxford-iiit-pet')
```
In Jupyter Notebook, you can just write a variable on its own (semicolon is just an end of statement in Python) and it prints it. You can also say `print(path)` but again, we are trying to do everything fast and interactively, so just write it and here is the path where it's given us our data. 

Next time you run this, since you've already downloaded it, it won't download it again. Since you've already untared it, it won't untar it again. So everything is designed to be pretty automatic and easy.

[[23:50](https://youtu.be/BWWm4AzsdLk?t=1430)]

There are some things in Python that are less convenient for interactive use than they should be. For example, when you do have a path object, seeing what's in it actually takes a lot more typing than I would like. So sometimes we add functionality into existing Python stuff. One of the things we do is add a `ls()` method to path.

```python
path.ls()
```
```
['annotations', 'images']
```

These are what's inside this path, so that's what we just downloaded. 

### Python 3 pathlib [[24:25](https://youtu.be/BWWm4AzsdLk?t=1465)]

```python
path_anno = path/'annotations'
path_img = path/'images'
```

If you are an experienced Python programmer, you may not be familiar with this approach of using a slash like this. This is a really convenient function that's part of Python 3. It's functionality from [pathlib](https://docs.python.org/3/library/pathlib.html). Path object is much better to use than strings. They let you use basically create sub paths like this. It doesn't matter if you're on Windows, Linux, or Mac. It is always going to work exactly the same way. `path_img` is the path to the images in that dataset.

[[24:57](https://youtu.be/BWWm4AzsdLk?t=1497)]

So if you are starting with a brand new dataset trying to do some deep learning on it. What do you do? Well, the first thing you would want to do is probably see what's in there. So we found that `annotations` and `images` are the directories in there, so what's in this images? 

### get_image_files [25:15](https://youtu.be/BWWm4AzsdLk?t=1515)

get_image_files will just grab an array of all of the image files based on extension in a path. 

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

 This is a pretty common way for computer vision datasets to get passed around - just one folder with a whole bunch of files in it. So the interesting bit then is how do we get the labels. In machine learning, the labels refer to the thing we are trying to predict. If we just eyeball this, we could immediately see that the labels are actually part of the file names. It's kind of like `path/label_number.extension`. We need to somehow get a list of `label` bits of each file name, and that will give us our labels. Because that's all you need to build a deep leaning model:
 - Pictures (files containing the images)
 - Labels

In fastai, this is made really easy. There is an object called `ImageDataBunch`. An ImageDataBunch represents all of the data you need to build a model and there's some factory method which try to make it really easy for you to create that data bunch - a training set, a validation set with images and labels. 

In this case, we need to extract the labels from the names. We are going to use `from_name_re`. `re` is the module in Python that does regular expressions - things that's really useful for extracting text. 

Here is the regular expression that extract the label for this dataset:

```
np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'
```
With this factory method, we can basically say:

- path_img: a path containing images
- fnames: a list of file names
- pat: a regular expression (i.e. pattern) to be used to extract the label from the file name
- ds_tfm: we'll talk about transforms later
- size: what size images do you want to work with.
  
This might seem weird because images have size. This is a shortcoming of current deep learning technology which is that a GPU has to apply the exact same instruction to a whole bunch of things at the same time in order to be fast. If the images are different shapes and sizes, you can't do that. So we actually have to make all of the images the same shape and size. In part 1 of the course, we are always going to be making images square shapes. Part 2, we will learn how to use rectangles as well. It turns out to be surprisingly nuanced. But pretty much everybody in pretty much all computer vision modeling nearly all of it uses this approach of square. 224 by 224, for reasons we'll learn about, is an extremely common size that most models tend to use so if you just use size=224, you're probably going to get pretty good results most of the time. This is kind of the little bits of artisanship that I want to teach you which is what generally just works. So if you just use size 224, that'll generally just work for most things most of the time.



```
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224)
data.normalize(imagenet_stats)
```

[29:16](https://youtu.be/BWWm4AzsdLk?t=1756)

`ImageDataBunch.from_name_re` is going to return a DataBunch object. In fastai, everything you model with is going to be a DataBunch object. Basically DataBunch object contains 2 or 3 datasets - it contains your training data, validation data, and optionally test data. For each of those, it contains your images and your labels, your texts and your labels, or your tabular data and your labels, or so forth. And that all sits there in this one place(i.e. `data`). 

Something we will learn more about in a little bit is normalization. But generally in nearly all machine learning tasks, you have to make all of your data about the same "size" - they are specifically about the same mean and standard deviation.  So there is a normalize function that we can use to normalize our data bunch in that way.



[30:25](https://youtu.be/BWWm4AzsdLk?t=1825)

Question: What does the function do if the image size is not 224? 

This is what we are going to learn about shortly. Basically this thing called transforms is used to do a number of the things and one of the things it does is to make something size 224. 


### data.show_batch
Let's take a look at a few pictures. Here are a few pictures of things from my data bunch. So you can see data.show_batch can be used to show me some of the contents in my data bunch. So you can see roughly what's happened is that they all seem to have being zoomed and cropped in a reasonably nice way. So basically what it'll do is something called by default center cropping which means it'll grab the middle bit and it'll also resize it. We'll talk more about the detail of this because it turns out to actually be quite important, but basically a combination of cropping and resizing is used.

```
data.show_batch(rows=3, figsize=(7,6))
```
![](lesson1/8.png)

Something else we are going to learn about is we also use this to do something called data augmentation. So there's actually some randomization in how much and where it crops and stuff like that. 

Basic the basic idea is some cropping, resizing, and padding. So there's all kinds of different ways it depends on data augmentation which we are going to learn about shortly. 


[31:51](https://youtu.be/BWWm4AzsdLk?t=1911)

Question: What does it mean to normalize the images? 

Normalizing the images, we're going to be learning more about later in the course, but in short, it means that the pixel values start out from naught to 255. And some channels might tend to be really bright, some might tend to be really not bright at all, some might vary a lot, and some might not very much at all. It really helps train a deep learning model if each one of those red green and blue channels has a mean of zero and a standard deviation of one. 

If your data is not normalized, it can be quite difficult for your model to train well. So if you have trouble training a model, one thing to check is that you've normalized it.


[33:00](https://youtu.be/BWWm4AzsdLk?t=1980)
Question: As GPU mem will be in power of 2, doesn't size 256 sound more practical considering GPU utilization compared to 224?

The brief answer is that the models are designed so that the final layer is of size 7 by 7, so we actually want something where if you go 7 times 2 a bunch of times (224 = 7*2*2*2*2*2), then you end up with something that's a good size.


[33:27](https://youtu.be/BWWm4AzsdLk?t=2007)

We will get to all these details but the key thing is I wanted to get to training a model as quickly as possible. 

### It is important to look at the data

One of the most important thing to be a really good practitioner is to be able to look at your data. So it's really important to remember to go `data.show_batch` and take a look. It's surprizing how often when you actually look at the dataset you've been given that you realize it's got weird black borders on it, some of the things have text covering up some of it, or some of it is rotated in odd ways. So make you take a look.

The other thing we want to do at is to look at the labels. All of the possible label names are called your classes. With DataBunch, you can print out your `data.classes`.

```
print(data.classes)
len(data.classes),data.c
```

```
['american_bulldog', 'german_shorthaired', 'japanese_chin', 'great_pyrenees', 'Bombay', 'Bengal', 'keeshond', 'shiba_inu', 'Sphynx', 'boxer', 'english_cocker_spaniel', 'american_pit_bull_terrier', 'Birman', 'basset_hound', 'British_Shorthair', 'leonberger', 'Abyssinian', 'wheaten_terrier', 'scottish_terrier', 'Maine_Coon', 'saint_bernard', 'newfoundland', 'yorkshire_terrier', 'Persian', 'havanese', 'pug', 'miniature_pinscher', 'Russian_Blue', 'staffordshire_bull_terrier', 'beagle', 'Siamese', 'samoyed', 'chihuahua', 'Egyptian_Mau', 'Ragdoll', 'pomeranian', 'english_setter']

(37, 37)
```

 That's all of the possible labels that we found by using that regular expression on the file names. We learnt earlier on at the top that there are 37 possible categories, so just checking `len(data.classes)`, it is indeed 37. DataBunch will always have a property called `c`. We will get to the technical detail later, but for now, you can kind of think of it as being the number of classes. For things like regression problems and multi-label classification, that's not exactly accurate, but it'll do for now. It is important to know that `data.c` is a really important piece of information that is something like, or at least for classification problems it is, the number of classes. 

 ## Training [[35:07](https://youtu.be/BWWm4AzsdLk?t=2107)]

 Believe it or not, we are now ready to train a model. A model is trained in fastai using something called a "learner". 
 
 - **DataBunch**: A general fastai concept for your data, and from there, there are subclasses for particular applications like ImageDataBunch
 - **Learner**: A general concept for things that can learn to fit a model. From that, there are various subclasses to make things easier in particular, there is one called ConvLearner (something that will create a convolutional neural network for you).

```
learn = ConvLearner(data, models.resnet34, metrics=error_rate)
```
  
For now, just know that to create a learner for a convolutional neural network, you just have to tell it two things:
`data`: What's your data. Not surprisingly, it takes a data bunch.
`arch`: What's your architecture. There are lots of different ways of constructing a convolutional neural network. 

For now, the most important thing for you to know is that there's a particular kind of model called ResNet which works extremely well nearly all the time. For a while, at least, you really only need to be doing choosing between two things which is what size ResNet do you want. There are ResNet34 and ResNet50. When we are getting started with something, I'll pick a smaller ome because it'll train faster. That's as much as you need to know to be a pretty good practitioner about architecture for now which is that there are two variants of one architecture that work pretty well: ResNet34 and ResNet50. Start with a smaller one and see if it's good enough.

That is all the information we need to create a convolutional neural network learner. 

There is one other thing I'm going to give it though which is a list of metrics. Metrics are literally just things that gets printed out as it's training. So I'm saying I would like you to print out error rate. 

[37:25](https://youtu.be/BWWm4AzsdLk?t=2245)

![](lesson1/c1.png)

The first time I run this on a newly installed box, it downloads the ResNet34 pre-trained weights. What that means is that this particular model has actually already been trained for a particular task. And that particular task is that it was trained on looking at about one and a half million pictures of all kinds of different things, a thousand categories of things, using an image dataset called ImageNet. So we can download those pre-trained weights so that we don't start with a model that knows nothing about anything, but we actually start with a model that knows how to recognize a thousand categories of things in ImageNet. I don't think all of these 37 categories of pet are in ImageNet but there were certainly some kinds of dog and some kinds of cat. So this pre-trained model knows quite a little bit about what pets look like, and it certainly knows quite a lot about what animals look like and what photos look like. So the idea is that we don't start with a model that knows nothing at all, but we start by downloading a model that knows something about recognizing images already. So it downloads for us automatically, the first time we use it, a pre-trained model and then from now on, it won't need to download it again - it'll just use the one we've got. 

## Transfer learning [38:54](https://youtu.be/BWWm4AzsdLk?t=2334)

This is really important. We are going to learn a lot about this. It's kind of the focus of the whole course which is how to do this thing called "transfer learning." How to take a model that already knows how to do something pretty well and make it so that it can do your thing really well. We will take a pre-trained model, and then we fit it so that instead of predicting a thousand categories of ImageNet with ImageNet data, it predicts the 37 categories of pets using your pet data. By doing this, you can train models in 1/100 or less of the time of regular model training with 1/100 or less of the data of regular model training. Potentially, many thousands of times less. Remember I showed you the slide of Nikhil's lesson 1 project from last year? He used 30 images. There are not cricket and baseball images in ImageNet but it turns out that ImageNet is already so good at recognizing things in the world that just 30 examples of people playing baseball and cricket was enough to build a nearly perfect classifier. 


## Validation set [40:05](https://youtu.be/BWWm4AzsdLk?t=2405)
