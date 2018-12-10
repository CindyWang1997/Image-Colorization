# colorNet-pytorch
A Neural Network For Automatic Image Colorization

This repository implements a neural network for image colorization for the final project of the class [Computer Vision](http://www.cs.columbia.edu/~vondrick/class/vision-fa18/). The code is adapted from the [PyTorch version](https://github.com/shufanwu/colorNet-pytorch) of the [ColorNet](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/) issued on SIGGRAPH 2016 AND the [Pytorch version](https://github.com/chuchienshu/Colorization) of [Colorful Image Colorization](https://arxiv.org/abs/1603.08511) issued in ECCV 2016.

## Overview
* DataSet
[MIT Places365-Standard](http://places2.csail.mit.edu/download.html)  


* Development Environment  
Python 3.5.2
CUDA 9.0  


* Net model<br />
Baseline model: this is the model used by Iizuka et al. which uses a two-stream architecture in which they fuse global and local features.
![...](https://github.com/CindyWang1997/Image-Colorization/blob/master/readme%20images/baseline_model.png)

Combined model: this is our proposed model, which is a combination of the model proposed by Iizauka et al. and Zhang et al. It substitutes the colorization network in Iizuka’s model with a network that maps combined features to probability distribution of 313 ab values bins. 
![...](https://github.com/CindyWang1997/Image-Colorization/blob/master/readme%20images/combined_model.png)


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
We trained both the baseline model and the proposed model for 3 epochs which is far fewer than those trained in both papers, so we expect it to work better if train it more.

* Results using our proposed model
![...](https://github.com/CindyWang1997/Image-Colorization/blob/master/readme%20images/colorization_results.png)  

* Comparison with baseline model<br />
With both models trained for 3 epochs, we note that when the image is anything otherthan outdoor natural sceneries, the baseline model gives dull uniform-colored outputs. Our model has some improvements in this aspect.
<img src="https://github.com/CindyWang1997/Image-Colorization/blob/master/readme%20images/baseline-comparison.png" width="500">

* Major limitation: unclear borders and color inconsistency<br />
Although out new model gives more vibrant colors to objects such as aircrafts, cars and other man-made objects, those colors are not very connsistent with the borders. Sometimes the vibrant colors (e.g. red of the car) extend outside the borders and other times it does not colorize the individual object entirely (e.g. the fish). This may be improved by adopting better segmentation methods and down/up sampling strategies.
![...](https://github.com/CindyWang1997/Image-Colorization/blob/master/readme%20images/color-inc.png)


