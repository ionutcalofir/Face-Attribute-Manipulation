# Face-Attribute-Manipulation

# Info
---
My tensorflow implementation of the paper [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)

# Results
---
Results after running the algorithm 1 epoch.

###### Note:
&nbsp;&nbsp; From left to right: original image, black hair, blond hair, male, female, young, old, smile.
<p><img src='images/results/image_0.png' /></p>
<p><img src='images/results/image_1.png' /></p>
<p><img src='images/results/image_2.png' /></p>
<p><img src='images/results/image_3.png' /></p>
<p><img src='images/results/image_4.png' /></p>
<p><img src='images/results/image_5.png' /></p>
<p><img src='images/results/image_6.png' /></p>

# Usage
Tensorflow: 1.7.0
CUDA: 9.0.176
cuDNN: 7.0.5

---
##### 1. Download dataset into ```./dataset```. It should be like this:
```
dataset
    CelebA
        anno
             original.txt
        imgs
            img1.jpg
            ...
            imgn.jpg
```
Where ```original.txt``` is a text file with images name and annotations.

##### 2. Run the following command to create the necessary folders and build dataset
```bash
$ python setup.py
```

##### 3. Main
##### &nbsp;&nbsp; Training
```bash
$ python main.py --train
```
##### &nbsp;&nbsp; Resume training
```bash
$ python main.py --train --resume
```
