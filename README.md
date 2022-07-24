# ProGAN
This repo is a pytorch implementation of 
[Progressive Growing of GANs for Improved Quality, Stability, and Variation (ProGAN)](https://arxiv.org/abs/1710.10196) on [celbA_hq dataset](https://www.kaggle.com/datasets/lamsimon/celebahq)

GANs, also known as Generative Adversarial Networks, are one of the most fascinating new developments in deep learning.
Yann LeCun saw GANs as "the most fascinating idea in the last 10 years in ML" when Ian Goodfellow and Yoshua Bengio from the University of Montreal first unveiled them in 2014.
GANS are frequently used to make deep fake films, improve the quality of images, face swap, design gaming characters, and much more. 

![](src/imgs/ProGANS.png)
Progressive Growing GAN also know as ProGAN introduced by Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen from NVIDIA and it is an extension of the training process of GAN that allows the generator models to train with stability which can produce large-high-quality images.

It involves training by starting with a very small image and then the blocks of layers added incrementally so that the output size of the generator model increases and increases the input size of the discriminator model until the desired image size is obtained. This type of approach has proven very effective at generating high-quality synthetic images that are highly realistic.

It basically involves 4 major steps

1) Progressive growing (of model and layers)

2) Minibatch std on Discriminator

3) Normalization with PixelNorm

4) Equalized Learning Rate

# Setup and Generate
This code is developed under following library dependencies
```commandline
python 3.8
torch 1.11.0
torchvision 0.12.0
```
Start with creating a virtual environment then open your terminal and follow the following steps:
```commandline
git clone "https://github.com/zaghlol94/ProGAN"
cd ProGAN
pip install -r requirements.txt
bash download_assets.sh
cd src
python generate.py
```
# Dataset and Train
## Dataset
CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes 
dataset with more than 200K celebrity images. 
```commandline
src
    └── celba_hq
        ├── train
        │   ├── female
        │   │   ├── 000001.jpg
        │   │   ├── 000002.jpg
        │   │   ├── .
        │   │   └── .
        │   └── male
        │       ├── 000001.jpg
        │       ├── 000002.jpg
        │       ├── .
        │       └── .
        └── val
            ├── female
            │   ├── 000001.jpg
            │   ├── 000002.jpg
            │   ├── .
            │   └── .
            └── male
                ├── 000001.jpg
                ├── 000002.jpg
                ├── .
                └── .

```
## Training
Download celaba_hq dataset and add the images' folder in ```src/celba_hq```
if you rename the root folder of the dataset don't forget to change the ````DATASET````
variable in [config.py](https://github.com/zaghlol94/ProGAN/blob/main/src/config.py)
```commandline
cd src
python train.py
```
after training, you could see the results of fake images in every step in tensorboard
```
tensorboard --logdir=logs/ 
```
# Results
![](src/imgs/64_examples.png)
# Citation
```commandline
@misc{https://doi.org/10.48550/arxiv.1710.10196,
  doi = {10.48550/ARXIV.1710.10196},
  
  url = {https://arxiv.org/abs/1710.10196},
  
  author = {Karras, Tero and Aila, Timo and Laine, Samuli and Lehtinen, Jaakko},
  
  keywords = {Neural and Evolutionary Computing (cs.NE), Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Progressive Growing of GANs for Improved Quality, Stability, and Variation},
  
  publisher = {arXiv},
  
  year = {2017},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```