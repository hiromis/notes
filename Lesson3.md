Lesson 3



Diagram Andrew Ng

coursera Machine Learning



Machine Learning fastai



Production area

- Deploying on Zeit - simple and free!
  - What Australian Car Is That Edward Ross 



Multi-label classification

lesson3-planet.ipynb



data block API



1. Dataset (PyTorch)
   1. `__getitem__` index by [] o[3]
   2. `__len__` len(o)
2. DataLoader - grabs individual items, combine them, and pop it on GPU
3. DataBunch - binds together train_dl, valid_dl



Datablock API

data_block.ipynb



canvid



metrics doesn't change how the model train



data.c - how many outputs do we want our model to create



accuracy_thresh

`partial` slightly customized version 



Question: When your model makes an incorrect prediction in a deployed app, is there a good way to “record” that error and use that learning to improve the model in a more targeted way?

Record it? You do it. Log? 



Good segway!



Question: Could someone talk a bit more about the data block ideology? I’m not quite sure how the blocks are meant to be used. Do they have to be in a certain order? Is there any other library that uses this type of programming I could look at?



ETL. 



Question: Video

webapi, grab the frame with web API. client side, OpenCV. 



just before it shoots up and go under 10. first

second part, frozen lr divide by 5 or 10. 



256. 



Lesson3-camvid.ipynb - Segmentation

medicine, life science, 



fastai dataset





Question: Is there a way to use learn.lr_find() and have it return a suggested number directly rather than having to plot it as a graph and then pick a learning rate by visually inspecting that graph?

No. Experiment! Not bottom. Going up. 



codes.txt - what the number means



valid.txt - 



tfm_y 



show_batch



BREAK



Question: unsupervised learning?

Question: 

"progressive resizing"



Question: 

acc_camvid



training loss > validation loss. Underfitting. Train for longer, last bit with lower learning rate. Decrease regularization. weight decay, dropout



For segmentation 



Unet! MICAI 3000 sitation

https://twitter.com/ORonneberger/status/1059816543561891840



learn.recorder.plot_losses()

plot_lr - fit one cycle



Jose Fernandez Portal



"Learning rate annealing"

Leslie Smith



One Hundred Layers Tiramisu



Running out of memory a lot.

Mixed precision training

half precision floating point

.to_fp16() when create learner. if kernel dies, old driver



lesson3-head-pose.ipynb



ImagePoints coordinates 

Regression model



IMDB

lesson3-imdb.ipynb

fastai.text



DataBunch.from_csv



Question: https://forums.fast.ai/t/lesson-3-in-class-discussion/29733/333



stochastic (with mini batch) gradient descent



SAGAR SHARMA Activation Functions: Neural Networks



Rectified Linear Unit = max(x, 0)



michael neilson



Universal approximation theorem



Question: tokenization San Francisco

https://forums.fast.ai/t/lesson-3-in-class-discussion/29733/358?u=hiromi

Image model, CNN, recurrent model RNN, 



Question: satellite image 4 channel pretrained

2 channel: create a third channel with 0 or average of other two channels. do it ahead of time and save or custom 



4 channel: modify the model itself. weight tensors . zeros or random versions. a couple more lessons.



Wrapping up.

- started out with it's easy to make web apps! Single label classifications
- multi-label classification such as planet. 
- segmentation 
- image regression
- NLP classifications and a lot more
- Gradient descent along with non linearity. Universal approximation theory tells

This week, see if you can come up with 



Next week, more NLP, SGD, Adam, RMSProp, webapps

























