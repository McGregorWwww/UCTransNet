# UCTransNet 


This repo is the official implementation of
["UCTransNet: Rethinking the Skip Connections in U-Net from
a Channel-wise Perspective with Transformer"]()

We propose a Channel Transformer module (CTrans) and use it to 
replace the skip connections in original U-Net, thus we name it "U-CTrans-Net".
## Requirements


Install from the ```requirement.txt``` using:
```angular2html
pip install -r requirements.txt
```

## Usage



### 1. Data Preparation
#### 1.1. GlaS and MoNuSeg Datasets
The original data can be downloaded in following links:
* MoNuSeG Dataset - [Link (Original)](https://monuseg.grand-challenge.org/Data/)
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
#### 3.1. Test the Model and Visualize the Segmentation Results
First, change the session name in ```Config.py``` as the training phase.
Then run:
```angular2html
python test_model.py
```
You can get the Dice and IoU scores and the visualization results. 
## Reference


* [TransUNet](https://github.com/Beckschen/TransUNet) 
* [MedT](https://github.com/jeya-maria-jose/Medical-Transformer)



## Citations


If this code is helpful for your study, please cite:
```
@misc{wang2021uctransnet,
      title={UCTransNet: Rethinking the Skip Connections in U-Net from a Channel-wise Perspective with Transformer}, 
      author={Haonan Wang and Peng Cao and Jiaqi Wang and Osmar R. Zaiane},
      year={2021},
      eprint={2109.04335},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Contact 
Haonan Wang ([haonan1wang@gmail.com](haonan1wang@gmail.com))
