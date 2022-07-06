# [AAAI2022] UCTransNet


This repo is the official implementation of
['UCTransNet: Rethinking the Skip Connections in U-Net from
a Channel-wise Perspective with Transformer'](https://ojs.aaai.org/index.php/AAAI/article/view/20144) which is accepted at AAAI2022.

![framework](https://github.com/McGregorWwww/UCTransNet/blob/main/docs/Framework.jpg)

We propose a Channel Transformer module (CTrans) and use it to 
replace the skip connections in original U-Net, thus we name it 'U-CTrans-Net'.


🔥🔥 **[Online Presentation Video](https://www.bilibili.com/video/BV1ZF411p7PM?spm_id_from=333.999.0.0) is available for brief introduction.** 🔥🔥

## Requirements

Install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```

## Usage



### 1. Data Preparation
#### 1.1. GlaS and MoNuSeg Datasets
The original data can be downloaded in following links:
* MoNuSeg Dataset - [Link (Original)](https://monuseg.grand-challenge.org/Data/)
* GLAS Dataset - [Link (Original)](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest)

Then prepare the datasets in the following format for easy use of the code:
```angular2html
├── datasets
    ├── GlaS
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    └── MoNuSeg
        ├── Test_Folder
        │   ├── img
        │   └── labelcol
        ├── Train_Folder
        │   ├── img
        │   └── labelcol
        └── Val_Folder
            ├── img
            └── labelcol
```
#### 1.2. Synapse Dataset
The Synapse dataset we used is provided by TransUNet's authors.
Please go to [https://github.com/Beckschen/TransUNet/blob/main/datasets/README.md](https://github.com/Beckschen/TransUNet/blob/main/datasets/README.md)
for details.

### 2. Training
As mentioned in the paper, we introduce two strategies 
to optimize UCTransNet.

The first step is to change the settings in ```Config.py```,
all the configurations including learning rate, batch size and etc. are 
in it.

#### 2.1 Jointly Training
We optimize the convolution parameters 
in U-Net and the CTrans parameters together with a single loss.
Run:
```angular2html
python train_model.py
```

#### 2.2 Pre-training

Our method just replaces the skip connections in U-Net, 
so the parameters in U-Net can be used as part of pretrained weights.

By first training a classical U-Net using ```/nets/UNet.py``` 
then using the pretrained weights to train the UCTransNet, 
CTrans module can get better initial features.

This strategy can improve the convergence speed and may 
improve the final segmentation performance in some cases.


### 3. Testing
#### 3.1. Get Pre-trained Models
Here, we provide pre-trained weights on GlaS and MoNuSeg, if you do not want to train the models by yourself, you can download them in the following links:
* GlaS：https://drive.google.com/file/d/1ciAwb2-0G1pZrt_lgSwd-7vH1STmxdYe/view?usp=sharing
* MoNuSeg: https://drive.google.com/file/d/1CJvHoh3VrPsBn_njZDo6SvJF_yAVe5MK/view?usp=sharing
#### 3.2. Test the Model and Visualize the Segmentation Results
First, change the session name in ```Config.py``` as the training phase.
Then run:
```angular2html
python test_model.py
```
You can get the Dice and IoU scores and the visualization results. 

🔥🔥 **The testing results of all classes in Synapse dataset can be downloaded through [this link](https://drive.google.com/file/d/1E-ZJLkNc0AJSUKI1CCWdcROMS9wERI9s/view?usp=sharing).** 🔥🔥


### 4. Reproducibility
In our code, we carefully set the random seed and set cudnn as 'deterministic' mode to eliminate the randomness. 
However, there still exsist some factors which may cause different training results, e.g., the cuda version, GPU types, the number of GPUs and etc. The GPU used in our experiments is NVIDIA A40 (48G) and the cuda version is 11.2.

Especially for multi-GPU cases, the upsampling operation has big problems with randomness.
See https://pytorch.org/docs/stable/notes/randomness.html for more details.

When training, we suggest to train the model twice to verify wheather the randomness is eliminated. Because we use the early stopping strategy, **the final performance may change significantly due to the randomness**. 

## Reference


* [TransUNet](https://github.com/Beckschen/TransUNet) 
* [MedT](https://github.com/jeya-maria-jose/Medical-Transformer)



## Citations


If this code is helpful for your study, please cite:
```
@article{UCTransNet,
	 title={UCTransNet: Rethinking the Skip Connections in U-Net from a Channel-Wise Perspective with Transformer}, 
	 volume={36}, 
	 url={https://ojs.aaai.org/index.php/AAAI/article/view/20144}, 
  DOI={10.1609/aaai.v36i3.20144},
	 number={3}, 
	 journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
	 author={Wang, Haonan and Cao, Peng and Wang, Jiaqi and Zaiane, Osmar R.}, 
	 year={2022}, 
	 month={Jun.}, 
	 pages={2441-2449}}
```


## Contact 
Haonan Wang ([haonan1wang@gmail.com](haonan1wang@gmail.com))
