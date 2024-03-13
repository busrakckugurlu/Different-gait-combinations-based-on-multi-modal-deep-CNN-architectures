# Different-gait-combinations-based-on-multi-modal-deep-CNN-architectures
## Description:
This repository contains the original codes of the article titled "Different gait combinations based on multi-modal deep CNN architectures".

![Project Image](https://github.com/busrakckugurlu/Different-gait-combinations-based-on-multi-modal-deep-CNN-architectures/blob/main/images/GEI_HConL.PNG)
## License:
If you find this repository useful, please cite this paper:
- Yaprak, B., Gedikli, E. Different gait combinations based on multi-modal deep CNN architectures. Multimed Tools Appl (2024). https://doi.org/10.1007/s11042-024-18859-9
## Requirements:
- py 3.9
- cudatoolkit=11.2
- cudnn=8.1
- tensorflow-gpu==2.5
- numpy==1.23.4
#### Datasets:
Download the gait datasets: CASIA-B([apply link](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)), Outdoor-Gait ([Baidu Yun](https://pan.baidu.com/s/1oW6u9olOZtQTYOW_8wgLow) with extract code (tjw0) OR [Google Drive](https://drive.google.com/drive/folders/1XRWq40G3Zk03YaELywxuVKNodul4TziG?usp=sharing))

#### Train:
- All training and test codes only for Mobilenet are presented in this repository. To perform the same training on other networks, namely VGG-16, Resnet-50, EfficientNet-B0, and ConvNext-base update the following two 
  lines in 'train_X.py' files of code for the relevant network.
  
For example:
> from tensorflow.keras.applications import MobileNet
> 
> base_model = MobileNet(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape),pooling="avg")

To:
> from tensorflow.keras.applications import EfficientNetB0
> 
> base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape),pooling="avg")
#### Test:
- During the testing phase of multi-modal networks, the load_testing_data functions in create_train_test_data.py and load_gallery_probe.py need to be renamed. For example like this, 'load_testing_data_gei' or 'load_testing_data_stack'.

#### Info:
- Parameter settings for all networks are as follows:
> SGD optimizer,
> learning rate of 0.0001,
> momentum of 0.9
  
- The optimum epoch numbers for all networks on two different Datasets are as follows:
  > CASIA-B:
  >> MobileNet - GEI:50epoch  	HConL:70epoch  	Head:80epoch 	 Leg:70epoch   SH:10epoch 	OF:7epoch
  > 
  >> EfficientNet - GEI:100epoch  	HConL:130epoch  	Head:140epoch 	 Leg:130epoch  	 SH:10epoch 	OF:10epoch
  > 
  >> ConvNext - GEI:12epoch  	HConL:20epoch  	Head:22epoch 	 Leg:22epoch  	 SH:7epoch 	OF:6epoch


  > Outdoor-Gait:
  >> MobileNet - GEI:15epoch  	HConL:11epoch  	Head:12epoch 	 Leg:12epoch  	 SH:8epoch 	
  > 
  >> EfficientNet - GEI:35epoch  	HConL:35epoch  	Head:37epoch 	 Leg:37epoch  	 SH:18epoch 
  > 
  >> ConvNext - GEI:10epoch  	HConL:11epoch  	Head:10epoch 	 Leg:10epoch  	 SH:11epoch 	

