# Different-gait-combinations-based-on-multi-modal-deep-CNN-architectures
## Description:
This repository contains the original codes of the article titled "Different gait combinations based on multi-modal deep CNN architectures".
## Requirements:
- py 3.9
- cudatoolkit=11.2
- cudnn=8.1
- tensorflow-gpu==2.5
- numpy==1.23.4
## Datasets:
- CASIA-B [url] (http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
- Outdoor-Gait [url] (https://www.sciencedirect.com/science/article/pii/S0031320319302912)

## Train:
- All training and test codes only for Mobilenet are presented in this repository. To perform the same training on other networks, namely VGG-16, Resnet-50, EfficientNet-B0, and ConvNext-base update the following two lines in training .py files of code for the relevant network.
- from tensorflow.keras.applications import MobileNet
- base_model = MobileNet(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape),pooling="avg")

For example:
- from tensorflow.keras.applications import EfficientNetB0
- base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape),pooling="avg")

Parameter settings for all networks are as follows:
- SGD optimizer
- learning rate of 0.0001
- momentum of 0.9
  
The optimum epoch numbers for all networks on two different Datasets are as follows:
- CASIA-B:
-   VGG16- GEI:  	HConL:  	Head: 	 Leg:  	 SH: 	OF:
  ResNet50- GEI:  	HConL:  	Head: 	 Leg:  	 SH: 	OF:
  MobileNet- GEI:  	HConL:  	Head: 	 Leg:  	 SH: 	OF:
  EfficientNet- GEI:  	HConL:  	Head: 	 Leg:  	 SH: 	OF:
  ConvNext- GEI:  	HConL:  	Head: 	 Leg:  	 SH: 	OF:


- Outdoor-Gait:
-   VGG16- GEI:  	HConL:  	Head: 	 Leg:  	 SH: 	OF:
  ResNet50- GEI:  	HConL:  	Head: 	 Leg:  	 SH: 	OF:
  MobileNet- GEI:  	HConL:  	Head: 	 Leg:  	 SH: 	OF:
  EfficientNet- GEI:  	HConL:  	Head: 	 Leg:  	 SH: 	OF:
  ConvNext- GEI:  	HConL:  	Head: 	 Leg:  	 SH: 	OF:


