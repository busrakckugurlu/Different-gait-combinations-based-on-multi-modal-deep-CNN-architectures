
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from collections import OrderedDict
import cv2
from skimage.io import imread, imshow, imread_collection, concatenate_images

src_dir= 'data/OutdoorGei/OutdoorGait_gei_train_test/gei'
dest_dir= 'data/HConL'

foot_src = 'data/Cropped_Leg'
head_src = 'data/Cropped_Head'


dir_mapping = OrderedDict([(src_dir, dest_dir)])




for dir, dest_dir in dir_mapping.items():
    print('Processing data in {}'.format(dir))
   
    for index, file_name in enumerate(os.listdir(dir)): 
        file_dir = os.path.join(dir, file_name)
        dest_file_dir = os.path.join(dest_dir, file_name)
        if not os.path.exists(dest_file_dir):
            os.mkdir(dest_file_dir)
            print(dest_file_dir, 'created')
#id
        for id in os.listdir(file_dir):
            id_dir = os.path.join(file_dir, id)
            dest_id_dir = os.path.join(dest_dir, file_name,id)
            if not os.path.exists(dest_id_dir):
                os.mkdir(dest_id_dir)
                
            foot_dir = os.path.join(foot_src, file_name,id)
            head_dir = os.path.join(head_src, file_name,id)
            
            for name in os.listdir(foot_dir):
                path1=os.path.join(foot_dir,name)
                path3=os.path.join(head_dir,name)
                
                img1 = cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(path3,cv2.IMREAD_GRAYSCALE)
                
                concat_img=cv2.hconcat([img1,img2])
                res_concat_img = cv2.resize(concat_img, (240, 240))
                
                frame_name=name
                frame_dest_dir = os.path.join(dest_id_dir,frame_name)
                cv2.imwrite(frame_dest_dir, res_concat_img)
            
            
   



           