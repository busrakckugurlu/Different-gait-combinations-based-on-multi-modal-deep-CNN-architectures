
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array 
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import random



def custom_generator(file_dir,input_shape,num_classes):
    

    data_dir = 'data/OutdoorGei/OutdoorGait_gei_train_test/gei/train'
    # data_dir = 'data/Cropped_Gei_head/train'
    # data_dir = 'data/Cropped_Gei_foot/train'
    # data_dir = 'data/HConL/train'
    
    frames_dir= file_dir
    
    training_x = []
    train_label = []
    
    x_shape = (1,)+input_shape
    y_shape = (1, num_classes)

    
    frameslist=[]
    with open(frames_dir) as fo:
        for line in fo:
            frameslist.append(line[:line.rfind(' ')])
    
    
    while(True):
        frameslist=shuffle(frameslist)
        for url in frameslist:
            
            training_x = np.zeros(x_shape)
            training_y = np.zeros(y_shape)
                
            img = load_img(os.path.join(data_dir, url), target_size=input_shape)
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            training_x[0]=gray_img_3channel
        
            label = int(url.split("\\")[1])-1
            training_y[0, label] = 1
            
            
            training_x= np.array(training_x)
            training_y = np.array(training_y)
        
            yield training_x, training_y
