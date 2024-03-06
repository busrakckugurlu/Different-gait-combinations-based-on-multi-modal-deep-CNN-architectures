
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.backend import set_session

from tools.create_train_test_data import load_training_data

x_train, y_train = load_training_data("data/Cropped_Leg")


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


base_model = MobileNet(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)),pooling="avg")
x = base_model.output
x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu')(x)
x = Dense(74, activation='softmax')(x)
net = Model(base_model.input, x)
net.summary()


for layer in net.layers:
   layer.trainable = True


# train
batch_size = 4
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=20, # 0. 
    width_shift_range=0.2, # 0.
    height_shift_range=0.2,# 0.
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    data_format=K.image_data_format())

net.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])
history=net.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train)/batch_size, epochs=70)
net.save('weights/MobileNet_Leg.ckpt') 


import pandas as pd 
hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = 'weights/.MobileNet_Leg.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)


