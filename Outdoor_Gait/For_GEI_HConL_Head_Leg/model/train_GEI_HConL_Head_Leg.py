
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
import numpy as np
from tensorflow.python.keras.backend import set_session


from tools.custom_generator import custom_generator


train_dir= 'tools/trainlist.txt'
steps_per_epoch=2481


num_classes=69
input_shape=(224,224,3)
epoch=11  ## for HConL

# epoch=15  ## for GEI
# epoch=12  ## for Head
# epoch=12  ## forLeg

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


base_model = MobileNet(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape),pooling="avg")
x = base_model.output
x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x) #69
net = Model(base_model.input, x)
net.summary()

for layer in net.layers:
   layer.trainable = True


net.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])

train_generator = custom_generator(train_dir,input_shape,num_classes)

history = net.fit_generator(train_generator,steps_per_epoch=steps_per_epoch, epochs = epoch) 

# net.save('weights/MobileNet_Outdoor_GEI.ckpt')                 
net.save('weights/MobileNet_Outdoor_HConL.ckpt') 
# net.save('weights/MobileNet_Outdoor_Leg.ckpt') 
# net.save('weights/MobileNet_Outdoor_Head.ckpt') 
         
import pandas as pd 
hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = 'weights/.MobileNet_Outdoor_HConL.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f) 
        
