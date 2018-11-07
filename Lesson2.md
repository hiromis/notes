# Lesson 2

[Video](https://youtu.be/Egp4Zajhzog) / [Lesson Forum]((https://forums.fast.ai/t/lesson-2-official-resources-and-updates/28630)) / [General Forum]((https://forums.fast.ai/t/faq-resources-and-official-course-updates/27934/))

## Deeper Dive into Computer Vision

Taking a deeper dive into computer vision applications, taking some of the amazing stuff you've all been doing during the week, and going even further.

### Forum tips and tricks [[0:17]](https://youtu.be/Egp4Zajhzog?t=17)

Two important forum topics:

- [FAQ, resources, and official course updates](https://forums.fast.ai/t/faq-resources-and-official-course-updates/27934/)

- [Lesson 2 official resources and updates](https://forums.fast.ai/t/lesson-2-official-resources-and-updates/28630)


#### "Summarize This Topic" [[2:32]](https://youtu.be/Egp4Zajhzog?t=152)

After just one week, the most popular thread has 1.1k replies which is intimidatingly large number. You shouldn't need to read all of it.  What you should do is click "Summarize This Topic" and it will only show the most liked ones.

![](lesson2/1.png)

####  Returning to work [[3:19]](https://youtu.be/Egp4Zajhzog?t=199)

https://course-v3.fast.ai/ now has a "Returning to work" section which will show you (for each specific platform you use):

- How to make sure you have the latest notebooks
- How to make sure you have the latest fastai library 

If things aren't working for you, if you get into some kind of messy situation, which we all do, just delete your instance and start again unless you've got mission-critical stuff there — it's the easiest way just to get out of a sticky situation.



### What people have been doing this week [[4:19]](https://youtu.be/Egp4Zajhzog?t=259)

[Share your work here](https://forums.fast.ai/t/share-your-work-here/27676/) 

![](lesson2/2.png)

- [Figuring out who is talking — is it Ben Affleck or Joe Rogan](https://forums.fast.ai/t/share-your-work-here/27676/143) 
- [Cleaning up Watsapp downloaded images folder to get rid of memes](https://forums.fast.ai/t/share-your-work-here/27676/97)



![](lesson2/3.png)

[Forum post](https://forums.fast.ai/t/share-your-work-here/27676/215)

One of the really interesting projects was looking at the sound data that was used in [this paper](https://arxiv.org/abs/1608.04363). In this paper, they were trying to figure out what kind of sound things were. They got a state of the art of nearly 80% accuracy. Ethan Sutin then tried using the lesson 1 techniques and got 80.5% accuracy, so I think this is pretty awesome. Best as we know, it's a new state of the art for this problem. Maybe somebody since has published something we haven't found it yet. So take all of these with a slight grain of salt, but I've mentioned them on Twitter and lots of people on Twitter follow me, so if anybody knew that there was a much better approach, I'm sure somebody would have said so.



[[6:01](https://youtu.be/Egp4Zajhzog?t=361)]

![](lesson2/4.png)

[Forum post](https://forums.fast.ai/t/share-your-work-here/27676/38)

Suvash has a new state of the art accuracy for Devanagari text recognition. I think he's got it even higher than this now. This is actually confirmed by the person on Twitter who created the dataset. I don't think he had any idea, he just posted here's a nice thing I did and this guy on Twitter said: "Oh, I made that dataset. Congratulations, you've got a new record." So that was pretty cool.



[6:28](https://youtu.be/Egp4Zajhzog?t=388)

![](lesson2/5.png)

[The Mystery of the Origin](https://medium.com/@alenaharley/the-mystery-of-the-origin-cancer-type-classification-using-fast-ai-libray-212eaf8d3f4e)

I really like this post from Alena Harley. She describes in quite a bit of detail about the issue of metastasizing cancers and the use of point mutations and why that's a challenging important problem. She's got some nice pictures describing what she wants to do with this and how she can go about turning this into pictures. This is the cool trick — it's the same with urning sounds into pictures and then using the lesson 1 approach. Here is turning point mutations into pictures and then using the lesson 1 approach. And it seems that she's got a new state of the art result by more than 30% beating the previous best. Somebody on Twitter who is a VP at a genomics analysis company looked at this as well and thought it looked to be a state of the art in this particular point mutation one as well. So that's pretty exciting. 

When we talked about last week this idea that this simple process is something which can take you a long way, it really can. I will mention that something like this one in particular is using a lot of domain expertise, like figuring out that picture to create. I wouldn't know how to do that because I don't really know what a point mutation is, let alone how to create something that visually is meaningful that a CNN could recognize. But the actual deep learning side is actually straight forward.



[[8:07](https://youtu.be/Egp4Zajhzog?t=487)]

![](lesson2/6.png)

Another cool result from Simon Willison and Natalie Downe, they created a cougar or not web application over the weekend and won the Science Hack Day award in San Francisco. So I think that's pretty fantastic. So lots of examples of people doing really interesting work. Hopefully this will be inspiring to you to think well to think wow, this is cool that I can do this with what I've learned. It can also be intimidating to think like wow, these people are doing amazing things. But it's important to realize that as thousands of people are doing this course, I'm just picking out a few of really amazing ones. And in fact Simon is one of these very annoying people like Christine Payne who we talked about last week who seems to be good at everything he does. He created Django which is the world's most popular web frameworks, he founded a very successful startup, etc. One of those annoying people who tends to keep being good at things, now turns out he's good at deep learning as well. So that's fine. Simon can go on and win a hackathon on his first week of playing with deep learning. Maybe it'll take you two weeks to win your first hackathon. That's okay. 



[[9:22](https://youtu.be/Egp4Zajhzog?t=562)]

![](lesson2/7.png)

I think it's important to mention this because there was this really inspiring blog post this week from James Dellinger who talked about how he created a bird classifier using techniques from lesson 1. But what I really found interesting was at the end, he said he nearly didn't start on deep learning at all because he went through the scikit-learn website which is one of the most important libraries of Python and he saw this. And he described in this post how he was just like that's not something I can do. That's not something I understand. Then this kind of realization of like oh, I can do useful things without reading the Greek, so I thought that was really cool message. 



[[10:01](https://youtu.be/Egp4Zajhzog?t=601)]

![](lesson2/8.png)

I really wanted to highlight Daniel Armstrong on the forum. I think really shows he's a great role model here. He was saying I want to contribute to the library and I looked at the docs and I just found it overwhelming. The next message, one day later, was I don't know what any of this is, I didn't know how much there is to it, caught me off guard, my brain shut down but I love the way it forces me to learn so much. And a day later, I just submitted my first pull request. So I think that's awesome. It's okay to feel intimidated. There's a lot. But just pick one piece and dig into it. Try and push a piece of code or a documentation update, or create a classifier or whatever.



[[10:49](https://youtu.be/Egp4Zajhzog?t=649)]

So here's lots of cool classifiers people have built. It's been really inspiring. 

- Trinidad and Tobago islanders versus masquerader classifier
- A zucchini versus cucumber classifier
- Dog and cat breed classifier from last week and actually doing some exploratory work to see what the main features were, and discovered that one was most hairy dog and naked cats. So there are interesting you can do with interpretation. 
- Somebody else in the forum took that and did the same thing for anime to find that they had accidentally discovered an anime hair color classifier.
- We can now detect the new versus the old Panamanian buses.
- Henri Palacci discovered that he can recognize with 85% accuracy which of 110 countries a satellite image is of which is definitely got to be beyond human performance of just about anybody. 
- Batik cloth classification with a hundred percent accuracy
- Dave Luo did this interesting one. He actually went a little bit further using some techniques we'll be discussing in the next couple of courses to build something that can recognize complete/incomplete/foundation buildings and actually plot them on aerial satellite view. 

So lots and lots of fascinating projects. So don't worry. It's only been one week. It doesn't mean everybody has to have had a project out yet. A lot of the folks who already have a project out have done a previous course, so they've got a bit of a head start. But we will see today how you can definitely create your own classifier this week. 



[[12:56]](https://youtu.be/Egp4Zajhzog?t=776)

![](lesson2/9.png)

So from today, after we did a bit deeper into really how to make these computer vision classifiers and particular work well, we're then going to look at the same thing for text. We're then going to look at the same thing for tabular data. They are more like spreadsheets and databases. Then we're going to look at collaborative filtering (i.e. recommendation systems). That's going to take us into a topic called embeddings which is a key underlying platform behind these applications. That will take us back into more computer vision and then back into more NLP. So the idea here is that it turns out that it's much better for learning if you see things multiple times so rather than being like okay, that's computer vision, you won't see it again for the rest of the course, we're actually going to come back to the two key applications NLP and computer vision a few weeks apart. That's going to force your brain to realize oh, I have to remember this. It's not must something I can throw away. 

[[14:06]](https://youtu.be/Egp4Zajhzog?t=846)

![](lesson2/10.png)

For people who have more of a hard sciences background in particular, a lot of folks find this hey, here's some code, type it in, start running it approach rather than here's lots of theory approach confusing and surprising and odd at first. So for those of you, I just wanted to remind you this basic tip which is keep going. You're not expected to remember everything yet. You're not expected to understand everything yet. You're not expected to know why everything works yet. You just want to be in a situation where you can enter the code and you can run it and you can get something happening and then you can start to experiment and you get a feel for what's going on. Then push on. Most of the people who have done the course and have gone on to be really successful watch the videos at least three times. So they kind of go through the whole lot and then go through it slowly the second time, then they go through it really slowly the third time. I consistently hear them say I get a lot more out of it each time I go through. So don't pause at lesson 1 and stop until you can continue. 

This approach is based on a lot of academic research into learning theory. One guy in particular David Perkins from Harvard has this really great analogy. He is a researcher into learning theory. He describes this approach of whole game which is basically if you're teaching a kid to play soccer, you don't first of all teach them about how the friction between a ball and grass works and then teach them how to saw a soccer ball with their bare hands, and then teach them the mathematics of parabolas when you kick something in the air. No. You say, here's a ball. Let's watch some people playing soccer. Okay, now we'll play soccer and then gradually over the following years, learn more and more so that you can get better and better at it. So this is kind of what we're trying to get you to do is to play soccer which in our case is to type code and look at the inputs and look at the outputs. 



## Teddy bear detector using Google Images [[16:21](https://youtu.be/Egp4Zajhzog?t=981)]

Let's dig into our first notebook which is called [lesson2-download.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb). What we are going to do is we are going to see how to create your own classifier with your own images. It's going to be a lot like last week's pet detector but it will detect whatever you like. So to be like some of those examples we just saw. How would you create your own Panama bus detector from scratch. This is approach is inspired by Adrian Rosebrock who has a terrific website called [pyimagesearch](https://www.pyimagesearch.com/) and he has this nice explanation of  [how to create a deep learning dataset using Google Images](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/). So that was definitely an inspiration for some of the techniques we use here, so thank you to Adrian and you should definitely check out his site. It's full of lots of good resources.

We are going to try to create a teddy bear detector. And we're going to separate teddy bears from black bears, from grizzly bears. This is very important. I have a three year old daughter and she needs to know what she's dealing with. In our house, you would be surprised at the number of monsters, lions, and other terrifying threats that are around particularly around Halloween. So we always need to be on the lookout to make sure that the things we're about to cuddle is in fact a genuine teddy bear. So let's deal with that situation as best as we can.

### Step 1: Gather URLs of each class of images

Our starting point is to find some pictures of teddy bears so we can learn what they look like. So I go to  https://images.google.com/ and I type in Teddy bear and I just scroll through until I find a goodly bunch of them. Okay, that looks like plenty of teddy bears to me.

Then I go back to [the notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb) and you can see it says "go to Google Images and search and scroll." The next thing we need to do is to get a list of all the URLs there. To do that, back in your google images, you hit <kbd>Ctrl</kbd><kbd>Shift</kbd><kbd>J</kbd> in Windows/Linux and <kbd>Cmd</kbd><kbd>Opt</kbd><kbd>J</kbd> in Mac, and you paste the following into the window that appears:

``` javascript
urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
```

![](lesson2/11.png)

This is a Javascript console for those of you who haven't done any Javascript before. I hit enter and it downloads my file for me. So I would call this teddies.txt and press "Save". Okay, now I have a file containing URLs of teddies. Then I would repeat that process for black bears and for grizzly bears, and I put each one in a file with an appropriate name. 



### Step 2: Download images [[19:39](https://youtu.be/Egp4Zajhzog?t=1179)]

So step 2 is we now need to download those URLs to our server. Because remember when we're using Jupyter Notebook, it's not running on our computer. It's running on SageMaker or Crestle, or Google cloud, etc. So to do that, we start running some Jupyer cells. Let's grab the fastai library:

```python
from fastai import *
from fastai.vision import *
```

And let's start with black bears. So I click on this cell for black bears and I'll run it. So here, I've got three different cells doing the same thing but different information. This is one way I like to work with Jupyter notebook. It's something that a lot of people with more strict scientific background are horrified by. This is not reproducible research. I click on the black bear cell, and run it to create a folder called black and a file called urls_black.txt for my black bears. I skip the next two cells.

```python
folder = 'black'
file = 'urls_black.txt'
```

```python
folder = 'teddys'
file = 'urls_teddys.txt'
```

```python
folder = 'grizzly'
file = 'urls_grizzly.txt'
```



 Then I run this cell to create that folder.

```python
path = Path('data/bears')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)
```



Then I go down to the next section and I run the next cell which is download images for black bears. So that's just going to download my black bears to that folder. 

```python
classes = ['teddys','grizzly','black']
```

```python
download_images(path/file, dest, max_pics=200)
```



Now I go back and I click on `'teddys'`. And I scroll back down and repeat the same thing. That way, I'm just going backwards and forwards to download each of the classes that I want. Very manual bur for me, I'm very iterative and very experimental, that work swell for me. If you are better at planning ahead than I am, you can write a proper loop or whatever and do it that way. But when you see my notebooks and see things that are kind of like configuration cells (i.e. doing the same thing in different places), this is a strong sign that I didn't run this in order. I clicked one place, went to another, ran that. For me, I'm experimentalist. I really like to experiment in my notebook, I treat it like a lab journal, I try things out and I see what happens. So this is how my notebooks end up looking. 



It's a really controversial topic. For a lot of people, they feel this is "wrong" that you should only ever run things top to bottom. Everything you do should be reproducible. For me, I don't think that's the best way of using human creativity. I think human creativity is best inspired by trying things out and seeing what happens and fiddling around. You can see how you go. See what works for you.



So that will download the images to your server. It's going to use multiple processes to do so. One problem there is if something goes wrong, it's a bit hard to see what went wrong. So you can see in the next section, there's a commented out section that says `max_workers=0`. That will do it without spinning up a bunch of processes and will tell you the errors better. So if things aren't downloading, try using the second version. 

```python
# If you have problems download, try with `max_workers=0` to see exceptions:
# download_images(path/file, dest, max_pics=20, max_workers=0)
```



### Step 3: Create ImageDataBunch [[22:50](https://youtu.be/Egp4Zajhzog?t=1370)]

The next thing that I found I needed to do was to remove the images that aren't actually images at all. This happens all the time. There's always a few images in every batch that are corrupted for whatever reason. Google image told us this URL had an image but it doesn't anymore. So we got this thing in the library called `verify_images` which will check all of the images in a path and will tell you if there's a problem. If you say `delete=True`, it will actually delete it for you. So that's a really nice easy way to end up with a clean dataset. 

```python
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_workers=8)
```

![](lesson2/12.png)

So at this point, I now have a bears folder containing a grizzly folder, teddys folder, and black folder. In other words, I have the basic structure we need to create an ImageDataBunch to start doing some deep learning. So let's go ahead and do that.

Now, very often, when you download a dataset from like Kaggle or from some academic dataset, there will often be folders called train, valid, and test containing the different datasets. In this case, we don't have a separate validation set because we just grabbed these images from Google search. But you still need a validation set, otherwise you don't know how well your model is going and we'll talk more about this in a moment. 

Whenever you create a data bunch, if you don't have a separate training and validation set, then you can just say the training set is in the current folder (i.e. `.` because by default, it looks in a folder called `train`) and I want you to set aside 20% of the data, please. So this is going to create a validation set for you automatically and randomly. You'll see that whenever I create a validation set randomly, I always set my random seed to something fixed beforehand. This means that every time I run this code, I'll get the same validation set.  In general, I'm not a fan of making my machine learning experiments reproducible (i.e. ensuring I get exactly the same results every time). The randomness is to me a really important part of finding out your is solution stable and it is going to work each time you run it. But what is important is that you always have the same validation set. Otherwise when you are trying to decide has this hyper parameter change improved my model but you've got a different set of data you are testing it on, then you don't know maybe that set of data just happens to be a bit easier. So that's why I always set the random seed here.

```python
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
```



[[25:37](https://youtu.be/Egp4Zajhzog?t=1537)]

We've now got a data bunch, so you can look inside at the `data.classes` and you'll see these are the folders that we created. So it knows that the classes (by classes, we mean all the possible labels) are black bear, grizzly bear, or teddy bear.



```python
data.classes
```

```
['black', 'grizzly', 'teddys']
```



We can run `show_batch` and take a little look. And it tells us straight away that some of these are going to be a little bit tricky.  Some of them are not photo, for instance. Some of them are cropped funny, if you ended up with a black bear standing on top of a grizzly bear, that might be tough. 

```python
data.show_batch(rows=3, figsize=(7,8))
```

![](lesson2/bears.png)

You can kind of double check here. Remember, `data.c` is the attribute which the classifiers tell us how many possible labels there are. We'll learn about some other more specific meanings of `c` later. We can see how many things are now training set, how many things are in validation set. So we've got 473 training set, 141 validation set.  

```python
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
```

```
(['black', 'grizzly', 'teddys'], 3, 473, 140)
```



### Step 4: Training a model [[26:49](https://youtu.be/Egp4Zajhzog?t=1609)]

So at that point, we can go ahead and create our convolutional neural network using that data. I tend to default to using a resnet34, and let's print out the error rate each time.

```python
learn = create_cnn(data, models.resnet34, metrics=error_rate)
```

Then run `fit_one_cycle` 4 times and see how we go. And we have a 2% error rate. So that's pretty good. Sometimes it's easy for me to recognize a black bear from a grizzly bear, but sometimes it's a bit tricky. This one seems to be doing pretty well.  

```python
learn.fit_one_cycle(4)
```

```
Total time: 00:54
epoch  train_loss  valid_loss  error_rate
1      0.710584    0.087024    0.021277    (00:14)
2      0.414239    0.045413    0.014184    (00:13)
3      0.306174    0.035602    0.014184    (00:13)
4      0.239355    0.035230    0.021277    (00:13)
```

After I make some progress with my model and things are looking good, I always like to save where I am up to to save me the 54 seconds of going back and doing it again. 

```python
learn.save('stage-1')
```

As per usual, we unfreeze the rest of our model. We are going to be learning more about what that means during the course. 

```python
learn.unfreeze()
```

Then we run the learning rate finder and plot it (it tells you exactly what to type). And we take a look.

```python
learn.lr_find()
```

```
LR Finder complete, type {learner_name}.recorder.plot() to see the graph.
```



We are going to be learning about learning rates today, but for now, here's what you need to know. On the learning rate finder, what you are looking for is the strongest downward slope that's kind of sticking around for quite a while. It's something you are going to have to practice with and get a feel for﹣which bit works. So if you are not sure which, try both learning rates and see which one works better. I've been doing this for a while and I'm pretty sure this (between 10^-5 and 10^-3) looks like where it's really learning properly, so I will probably pick something back here for my learning rate [[28:28](https://youtu.be/Egp4Zajhzog?t=1708)].

```python
learn.recorder.plot()
```

![](lesson2/13.png)

So you can see, I picked `3e-5` for my bottom learning rate. For my top learning rate, I normally pick 1e-4 or 3e-4, it's kind of like I don't really think about it too much. That's a rule of thumb﹣it always works pretty well. One of the things you'll realize is that most of these parameters don't actually matter that much in detail. If you just copy the numbers that I use each time, the vast majority of the time, it'll just work fine. And we'll see places where it doesn't today.

```python
learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))
```

```
Total time: 00:28
epoch  train_loss  valid_loss  error_rate
1      0.107059    0.056375    0.028369    (00:14)
2      0.070725    0.041957    0.014184    (00:13)
```

So we've got 1.4% error rate after doing another couple of epochs, so that's looking great. So we've downloaded some images from Google image search, created a classifier, and we've got 1.4% error rate, let's save it.

```python
learn.save('stage-2')
```



### Interpretation [[29:38](https://youtu.be/Egp4Zajhzog?t=1778)]

As per usual, we can use the ClassificationInterpretation class to have a look at what's going on.

```python
learn.load('stage-2')
```

```python
interp = ClassificationInterpretation.from_learner(learn)
```

```python
interp.plot_confusion_matrix()
```

![](lesson2/14.png)

In this case, we made one mistake. There was one black bear classified as grizzly bear. So that's a really good step. We've come a long way. But possibly you could do even better if your dataset was less noisy. Maybe Google image search didn't give you exactly the right images all the time. So how do we fix that? We want to clean it up. So combining human expert with a computer learner is a really good idea. Very very few people publish on this or teach this, but to me, it's the most useful skill, particularly for you. Most of the people watching this are domain experts, not computer science experts, so this is where you can use your knowledge of point mutations in genomics or Panamanian buses or whatever. So let's see how that would work. What I'm going to do is, do you remember the plot top losses from last time where we saw the images which it was either the most wrong about or the least confident about. We are going to look at those and decide which of those are noisy. If you think about it, it's very unlikely that if there is a mislabeled data that it's going to be predicted correctly and with high confidence. That's really unlikely to happen. So we're going to focus on the ones which the model is saying either it's not confident of or it was confident of and it was wrong about. They are the things which might be mislabeled. 

A big shout-out to the San Francisco fastai study group who created this new widget this week called the FileDeleter. Zach, Jason, and Francisco built this thing where we basically can take the top losses from that interpretation object we just created. There is not just `plot_top-losses` but there's also `top_losses` and top_losses returns two things: the losses of the things that were the worst and the indexes into the dataset of the things that were the worst. If you don't pass anything at all, it's going to actually return the entire dataset, but sorted so the first things will be the highest losses. Every dataset in fastai has `x` and `y` and the `x` contains the things that are used to, in this case, get the images. So this is the image file names and the `y`'s will be the labels. So if we grab the indexes and pass them into the dataset's `x`, this is going to give us the file names of the dataset ordered by which ones had the highest loss (i.e. which ones it was either confident and wrong about or not confident about). So we can pass that to this new widget that they've created.

Just to clarify, this `top_loss_paths` contains all of the file names in our dataset. When I say "out dataset", this particular one is our validation dataset. So what this is going to do is it's going to clean up mislabeled images or images that shouldn't be there and we're going to remove them from a validation set so that our metrics will be more correct. You then need to rerun these two steps replacing `valid_ds` with `train_ds` to clean up your training set to get the noise out of that as well. So it's a good practice to do both. We'll talk about test sets later as well, if you also have a test set, you would then repeat the same thing. 

```python
from fastai.widgets import *

losses,idxs = interp.top_losses()
top_loss_paths = data.valid_ds.x[idxs]
```

```python
fd = FileDeleter(file_paths=top_loss_paths)
```

![](/Users/hiromi/git/notes/lesson2/16.png)

So we run FileDeleter passing in that sorted list of paths and so what pops up is basically the same thing as `plot_top_losses`. In other words, these are the ones which is either wrong about or least confident about. So not surprisingly, this one her (the second from left) does not appear to be a teddy bear, black bear, or grizzly bear. So this shouldn't be in our dataset. So what I do is I wack on the delete button, all the rest do look indeed like bears, so I can click confirm and it'll bring up another five. 

What I tend to do when I do this is I'll keep going confirm until I get to a coupe of screen full of the things that all look okay and that suggests to me that I've got past the worst bits of the data. So that's it so now you can go back for the training set as well and retrain your model. 

I'll just note here that what our San Francisco study group did here was that they actually built a little app inside Jupyter notebook which you might not have realized as possible. But not only is it possible, it's actually surprisingly straightforward. Just like everything else, you can hit double question mark to find out their secrets. So here is the source code. 

![](lesson2/17.png)

Really, if you've done any GUI programming before, it'll look incredibly normal. There's basically call backs for what happens when you click on a button where you just do standard Python things and to actually render it, you just use widgets and you can lay it out using standard boxes. So this idea of creating applications inside notebooks is really underused but it's super neat because it lets you create tools for your fellow practitioners or experimenters. And you could definitely envisage taking this a lot further. In fact, by the time you're watching this on the MOOC, you will probably find that there's a whole a lot more buttons here because we've already got a long list of to-do that we're going to add to this particular thing. 

I'd love for you to have a think about, now that you know it's possible to write applications in your notebook, what are you going to write and if you google for "[ipywidgets](https://ipywidgets.readthedocs.io/en/stable/)", you can learn about the little GUI framework to find out what kind of widgets you can create, what they look like, and how they work, and so forth. You'll find it's actually a pretty complete GUI programming environment you can play with. And this will all work nice with your models. It's not a great way to productionize an application because it is sitting inside a notebook. This is really for things which are going to help other practitioners or experimentalists. For productionizing things, you need to actually build a production web app which we will look at next. 

### Putting your model in production [[37:36](https://youtu.be/Egp4Zajhzog?t=2256)]

After you have cleaned up your noisy images, you can then retrain your model and hopefully you'll find it's a little bit more accurate. One thing you might be interested to discover when you do this is it actually doesn't matter most of the time very much. On the whole, these models are pretty good at dealing with moderate amounts of noisy data. The problem would occur is if your data was not randomly noisy but biased noisy. So I guess the main thing I'm saying is if you go through this process of cleaning up your data and then rerun your model and find it's .001% better, that's normal. It's fine. But it's still a good idea just to make sure that you don't have too much noise in your data in case it is biased.

At this point, we're ready to put our model in production and this is where I hear a lot of people ask me about which mega Google Facebook highly distributed serving system they should use and how do they use a thousand GPUs at the same time. For the vast majority of things you all do, you will want to actually run in production on a CPU, not a GPU. Why is that? Because GPU is good at doing lots of things at the same time, but unless you have a very busy website, it's pretty unlikely that you're going to have 64 images to classify at the same time to put into a batch into a GPU. And if you did, you've got to deal with all that queuing and running it all together, all of your users have to wait until that batch has got filled up and run﹣it's whole a lot of hassle. Then if you want to scale that, there's another whole lot of hassle. It's much easier if you just wrap one thing, throw it at a CPU to get it done, and comes back again. Yes, it's going to take maybe 10 or 20 times longer so maybe it'll take 0.2 seconds rather than 0.01 seconds. That's about the kind of times we are talking about. But it's so easy to scale. You can chuck it on any standard serving infrastructure. It's going to be cheap, and you can horizontally scale it really easily.  So most people I know who are running apps that aren't at Google scale, based on deep learning are using CPUs. And the term we use is "inference". When you are not training a model but you've got a trained model and you're getting it to predict things, we call that inference. That's why we say here:

> You probably want to use CPU for inference



At inference time, you've got your pre-trained model, you saved those weights, and how are you going to use them to create something like Simon Willison's cougar detector?

The first thing you're going to need to know is what were the classes that you trained with. You need to know not just what are they but what were the order. So you will actually need to serialize that or just type them in, or in some way make sure you've got exactly the same classes that you trained with. 

```python
data.classes
```

```
['black', 'grizzly', 'teddys']
```

If you don't have a GPU on your server, it will use the CPU automatically. If you have a GPU machine and you want to test using a CPU, you can just uncomment this line and that tells fastai that you want to use CPU by passing it back to PyTorch.  

```python
# fastai.defaults.device = torch.device('cpu')
```



[[41:14](https://youtu.be/Egp4Zajhzog?t=2474)]

So here is an example. We don't have a cougar detector, we have a teddy bear detector. And my daughter Claire is about to decide whether to cuddle this friend. What she does is she takes daddy's deep learning model and she gets a picture of this and here is a picture that she's uploaded to the web app and here is a picture of the potentially cuddlesome object. We are going to store that in a variable called `img` , and open_image is how you open an image in fastai, funnily enough.

```python
img = open_image(path/'black'/'00000021.jpg')
img
```

![](lesson2/bear.png)

Here is that list of classes that we saved earlier. And as per usual, we created a data bunch, but this time, we're not going to create a data bunch from a folder full of images, we're going to create a special kind of data bunch which is one that's going to grab one single image at a time. So we're not actually passing it any data. The only reason we pass it a path is so that it knows where to load our model from. That's just the path that's the folder that the model is going to be in. 

But what we need to do is that we need to pass it the same information that we trained with. So the same transforms, the same size, the same normalization. This is all stuff we'll learn more about. But just make sure it's the same stuff that you used before. 

Now you've got a data bunch that actually doesn't have any data in it at all. It's just something that knows how to transform a new image in the same way that you trained with so that you can now do inference. 

You can now `create_cnn` with this kind of fake data bunch and again, you would use exactly the same model that you trained with. You can now load in those saved weights. So this is the stuff that you only do once﹣just once when your web app is starting up. And it takes 0.1 of a second to run this code.

```python
classes = ['black', 'grizzly', 'teddys']
data2 = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(data2, models.resnet34)
learn.load('stage-2')
```

Then you just go `learn.predict(img)` and it's lucky we did that because it's not a teddy bear. This is actually a black bear. So thankfully due to this excellent deep learning model, my daughter will avoid having a very embarrassing black bear cuddle incident. 

```python
pred_class,pred_idx,outputs = learn.predict(img)
pred_class
```

```
'black'
```

So what does this look like in production? I took [Simon Willison's code](https://github.com/simonw/cougar-or-not), shamelessly stole it, made it probably a little bit worse, but basically it's going to look something like this. Simon used a really cool web app toolkit called [Starlette](https://www.starlette.io/). If you've ever used Flask, this will look extremely similar but it's kind of a more modern approach﹣by modern what I really mean is that you can use `await` which is basically means that you can wait for something that takes a while, such as grabbing some data, without using up a process. So for things like I want to get a prediction or I want to load up some data, it's really great to be able to use this modern Python 3 asynchronous stuff. So Starlette could come highly recommended for creating your web app.   

You just create a route as per usual, in that you say this is `async` to ensure it doesn't steal the process while it's waiting for things. 

You open your image you call `learner.predict`  and you return that response. Then you can use Javascript client or whatever to show it. That's it. That's basically the main contents of your web app.  

```python
@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    img = open_image(BytesIO(bytes))
    _,_,losses = learner.predict(img)
    return JSONResponse({
        "predictions": sorted(
            zip(cat_learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })
```

So give it a go this week. Even if you've never created a web application before, there's a lot of nice little tutorials online and kind of starter code, if in doubt, why don't you try Starlette. There's a free hosting that you can use, there's one called [PythonAnywhere](https://www.pythonanywhere.com/), for example. The one Simon has used, [Zeit Now](https://zeit.co/now), it's something you can basically package it up as a docker thing and shoot it off and it'll serve it up for you. So it doesn't even need to cost you any money and all these classifiers that you're creating, you can turn them into web application. I'll be really interested to see what you're able to make of that. That'll be really fun. 

### Things that can go wrong [[46:06](https://youtu.be/Egp4Zajhzog?t=2766)]

I mentioned that most of the time, the kind of rules of thumb I've shown you will probably work. And if you look at the share your work thread, you'll find most of the time, people are posting things saying I downloaded these images, I tried this thing, they worked much better than I expected, well that's cool. Then like 1 out of 20 says I had a problem. So let's have a talk about what happens when you have a problem. This is where we start getting into a little bit of theory because in order to understand why we have these problems and how we fix them, it really helps to know a little bit about what's going on.

First of all, let's look at examples of some problems. The problems basically will be either:

- Your learning rate is too high or low
- Your number of epochs is too high or low 

So we are going to learn about what those mean and why they matter. But first of all, because we are experimentalists, let's try them. 





### Learning rate (LR) too high



```python
learn = create_cnn(data, models.resnet34, metrics=error_rate)
```



```python
learn.fit_one_cycle(1, max_lr=0.5)
```

```
Total time: 00:13
epoch  train_loss  valid_loss  error_rate       
1      12.220007   1144188288.000000  0.765957    (00:13)
```



### Learning rate (LR) too low



```python
learn = create_cnn(data, models.resnet34, metrics=error_rate)
```



Previously we had this result:

```
Total time: 00:57
epoch  train_loss  valid_loss  error_rate
1      1.030236    0.179226    0.028369    (00:14)
2      0.561508    0.055464    0.014184    (00:13)
3      0.396103    0.053801    0.014184    (00:13)
4      0.316883    0.050197    0.021277    (00:15)
```



```python
learn.fit_one_cycle(5, max_lr=1e-5)
```

```
Total time: 01:07
epoch  train_loss  valid_loss  error_rate
1      1.349151    1.062807    0.609929    (00:13)
2      1.373262    1.045115    0.546099    (00:13)
3      1.346169    1.006288    0.468085    (00:13)
4      1.334486    0.978713    0.453901    (00:13)
5      1.320978    0.978108    0.446809    (00:13)
```



```python
learn.recorder.plot_losses()
```

![](lesson2/15.png)



As well as taking a really long time, it's getting too many looks at each image, so may overfit.



### Too few epochs



```python
learn = create_cnn(data, models.resnet34, metrics=error_rate, pretrained=False)
```



```python
learn.fit_one_cycle(1)
```

```
Total time: 00:14
epoch  train_loss  valid_loss  error_rate
1      0.602823    0.119616    0.049645    (00:14)
```



### Too many epochs



```python
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.9, bs=32, 
        ds_tfms=get_transforms(do_flip=False, max_rotate=0, max_zoom=1, max_lighting=0, max_warp=0
                              ),size=224, num_workers=4).normalize(imagenet_stats)
```



```python
learn = create_cnn(data, models.resnet50, metrics=error_rate, ps=0, wd=0)
learn.unfreeze()
```



```python
learn.fit_one_cycle(40, slice(1e-6,1e-4))
```

```
Total time: 06:39
epoch  train_loss  valid_loss  error_rate
1      1.513021    1.041628    0.507326    (00:13)
2      1.290093    0.994758    0.443223    (00:09)
3      1.185764    0.936145    0.410256    (00:09)
4      1.117229    0.838402    0.322344    (00:09)
5      1.022635    0.734872    0.252747    (00:09)
6      0.951374    0.627288    0.192308    (00:10)
7      0.916111    0.558621    0.184982    (00:09)
8      0.839068    0.503755    0.177656    (00:09)
9      0.749610    0.433475    0.144689    (00:09)
10     0.678583    0.367560    0.124542    (00:09)
11     0.615280    0.327029    0.100733    (00:10)
12     0.558776    0.298989    0.095238    (00:09)
13     0.518109    0.266998    0.084249    (00:09)
14     0.476290    0.257858    0.084249    (00:09)
15     0.436865    0.227299    0.067766    (00:09)
16     0.457189    0.236593    0.078755    (00:10)
17     0.420905    0.240185    0.080586    (00:10)
18     0.395686    0.255465    0.082418    (00:09)
19     0.373232    0.263469    0.080586    (00:09)
20     0.348988    0.258300    0.080586    (00:10)
21     0.324616    0.261346    0.080586    (00:09)
22     0.311310    0.236431    0.071429    (00:09)
23     0.328342    0.245841    0.069597    (00:10)
24     0.306411    0.235111    0.064103    (00:10)
25     0.289134    0.227465    0.069597    (00:09)
26     0.284814    0.226022    0.064103    (00:09)
27     0.268398    0.222791    0.067766    (00:09)
28     0.255431    0.227751    0.073260    (00:10)
29     0.240742    0.235949    0.071429    (00:09)
30     0.227140    0.225221    0.075092    (00:09)
31     0.213877    0.214789    0.069597    (00:09)
32     0.201631    0.209382    0.062271    (00:10)
33     0.189988    0.210684    0.065934    (00:09)
34     0.181293    0.214666    0.073260    (00:09)
35     0.184095    0.222575    0.073260    (00:09)
36     0.194615    0.229198    0.076923    (00:10)
37     0.186165    0.218206    0.075092    (00:09)
38     0.176623    0.207198    0.062271    (00:10)
39     0.166854    0.207256    0.065934    (00:10)
40     0.162692    0.206044    0.062271    (00:09)
```







Putting your model in production

CPU for inference

Claire 



Starlette



Python anywhere



BREAK

clean up validation set and test set

ipywidget





When you have a problem



`learn.recorder.plot_losses()`

Training loss should never be higher than validation loss. # epoch too low or LR too low



don't compare train_loss to valid loss to check overfitting

error starts getting worse.





argmax



Question:  error_Rate



Question: Why 3e-3 good learning rate before unfreezing. then 3e-4. min is out of learning rate finder.





y = ax + b



kahn academy is good



y = a1x + a2



y = a1x1 + a2x2  x2=1



coefficient



dot product

matrix product



Question: How many images are enough?

good learning rage, trained long enough, still not happy with accuracy. get more data. 



Question: Unbalanced classes?

Try it. it works. 



Question: 

do another cycle, train a few more cycles. 



Question: https://forums.fast.ai/t/lesson-2-chat/28722/139





PyTorch doesn't really like loops



https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-sgd.ipynb





`x@a`



regular shaped array

not jagged array

image is 3D tensor

Rank, or axis 



all rows, column 0

uniform_



plt matplotlib



50 million numbers



MSE



elementwise arithmetic 



Derivative



Quadratic



that's why we need good learning rate



Mini batches



Vocab



- Learning rate

- Epoch one complete run through all of our data points
- Minibatch
- SGD
- Model / Architecture
- Parameters : coefficients, weights
- Loss function : How far away/closer to 





You're a math person

There's no such thing as "not a math person"

[There’s no such thing as “not a math person”](https://www.youtube.com/watch?v=q6DGVGJ1WP4)

 

Underfitting "Just right" Overfitting

Validation set

How (and why) to create a good validation set



Build web application

