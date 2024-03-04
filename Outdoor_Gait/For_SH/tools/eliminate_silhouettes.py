
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from collections import OrderedDict
import cv2
from skimage.io import imread, imshow, imread_collection, concatenate_images

src_dir= 'data/OutdoorGei/OutdoorGait_test/test/segmentations'
dest_dir= 'data/Siluet_elimineted/test'

# src_dir= 'data/OutdoorGei/OutdoorGait_train/train/segmentations'
# dest_dir= 'data/Siluet_elimineted/train'

dir_mapping = OrderedDict([(src_dir, dest_dir)])


for dir, dest_dir in dir_mapping.items():
    print('Processing data in {}'.format(dir))
   
    for index, file_name in enumerate(os.listdir(dir)):
        file_dir = os.path.join(dir, file_name)
        dest_file_dir = os.path.join(dest_dir, file_name)
        if not os.path.exists(dest_file_dir):
            os.mkdir(dest_file_dir)
            print(dest_file_dir, 'created')
        for scene in os.listdir(file_dir):
            scene_dir = os.path.join(file_dir, scene)
            dest_scene_dir = os.path.join(dest_dir, file_name,scene)
            if not os.path.exists(dest_scene_dir):
                os.mkdir(dest_scene_dir)
        
            for name in os.listdir(scene_dir):
                path=os.path.join(scene_dir,name)
                
                img = cv2.imread(path)
                white_pixs = np.sum(img == 255)
                total_pixs = img.shape[0]*img.shape[1]
                if white_pixs/total_pixs <= 0.2:
                    continue   
                
                frame_name=name
                frame_dest_dir = os.path.join(dest_scene_dir,frame_name)
                cv2.imwrite(frame_dest_dir, img)