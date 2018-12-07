# colorNet-pytorch
A Neural Network For Automatic Image Colorization

This repository implements a neural network for image colorization for the final project of the class [Computer Vision](http://www.cs.columbia.edu/~vondrick/class/vision-fa18/). The code is adapted from the [PyTorch version](https://github.com/shufanwu/colorNet-pytorch) of the [ColorNet](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/) issued on SIGGRAPH 2016 AND the [Pytorch version](https://github.com/chuchienshu/Colorization) of [Colorful Image Colorization](https://arxiv.org/abs/1603.08511) issued in ECCV 2016.

## Overview
* DataSet
[MIT Places365-Standard](http://places2.csail.mit.edu/download.html)  


* Development Environment  
Python 3.5.2
CUDA 9.0  


* Net model
Baseline model: this is the model used by Iizuka et al. which uses a two-stream architecture in which they fuse global and local features.
![...](https://github.com/CindyWang1997/Image-Colorization/blob/master/readme%20images/model.png)



## Training and Validation
training from scratch:
```shell
# baseline
python3 train_baseline.py 
# new model
python3 train.py 
```
To resume training from a previous model, simply place the model under pretrained/ and run the same command.

testing and save output images:
```shell
# baseline
python3 val_baseline.py
# new model
python3 val.py
```

## Result
I just train this model for 3 epochs while 11 epochs in the paper， so I think it will work better if train it more.

* Good results  
![...](https://github.com/shufanwu/colorNet-pytorch/blob/master/readme%20images/good-result.png)  
* Bad results  
![...](https://github.com/shufanwu/colorNet-pytorch/blob/master/readme%20images/bad-result.png)  
For this network is trained by landscape image database, it's work well for scenery pictures. So if you use this network to color  images of other types, maybe you can't get a satisfying output.


