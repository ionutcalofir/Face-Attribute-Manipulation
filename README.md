# Face-Attribute-Manipulation

# Info
---
My tensorflow implementation of the paper [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)

# Usage
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
