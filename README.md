# Face-Attribute-Manipulation

# Info
---
My tensorflow implementation of the paper [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)

# Results
---
Results after running the algorithm 1 epoch.

###### Note:
&nbsp;&nbsp; From left to right: original image, black hair, blond hair, male, female, young, old, smile.
<p><img src='images/results/p_final_0.png' /></p>
<p><img src='images/results/p_final_1.png' /></p>
<p><img src='images/results/p_final_2.png' /></p>
<p><img src='images/results/p_final_3.png' /></p>
<p><img src='images/results/p_final_4.png' /></p>
<p><img src='images/results/p_final_5.png' /></p>
<p><img src='images/results/p_final_6.png' /></p>
<p><img src='images/results/p_final_7.png' /></p>
<p><img src='images/results/p_final_8.png' /></p>
<p><img src='images/results/p_final_9.png' /></p>
<p><img src='images/results/p_final_10.png' /></p>
<p><img src='images/results/p_final_11.png' /></p>
<p><img src='images/results/p_final_12.png' /></p>
<p><img src='images/results/p_final_13.png' /></p>
<p><img src='images/results/p_final_14.png' /></p>
<p><img src='images/results/p_final_15.png' /></p>
<p><img src='images/results/p_final_16.png' /></p>
<p><img src='images/results/p_final_17.png' /></p>
<p><img src='images/results/p_final_18.png' /></p>
<p><img src='images/results/p_final_19.png' /></p>

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
