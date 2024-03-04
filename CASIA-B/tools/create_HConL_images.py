
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from collections import OrderedDict
import cv2
from skimage.io import imread, imshow, imread_collection, concatenate_images

src_dir= 'data/Casia_gait/gei'
dest_dir= 'data/HConL'

foot_src = 'data/Cropped_Leg'
head_src = 'data/Cropped_Head'


dir_mapping = OrderedDict([(src_dir, dest_dir)])


for dir, dest_dir in dir_mapping.items():
    print('Processing data in {}'.format(dir))
#ids    
    for index, class_name in enumerate(os.listdir(dir)): 
        class_dir = os.path.join(dir, class_name)
        dest_class_dir = os.path.join(dest_dir, class_name)
        if not os.path.exists(dest_class_dir):
            os.mkdir(dest_class_dir)
            print(dest_class_dir, 'created')
#categories       
        for category in os.listdir(class_dir):
            ctg_dir = os.path.join(class_dir, category)
            dest_ctg_dir = os.path.join(dest_dir, class_name,category) 
            if not os.path.exists(dest_ctg_dir):
                os.mkdir(dest_ctg_dir)
            
            foot_dir = os.path.join(foot_src, class_name,category)
            head_dir = os.path.join(head_src, class_name,category)
            
        
            for name in os.listdir(foot_dir):
                path1=os.path.join(foot_dir,name)
                path3=os.path.join(head_dir,name)
                
                img1 = cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(path3,cv2.IMREAD_GRAYSCALE)
                
                print(name)
                
                concat_img=cv2.hconcat([img1,img2])
                res_concat_img = cv2.resize(concat_img, (240, 240))
                # imshow(res_concat_img)
                                     
                frame_name=name
                frame_dest_dir = os.path.join(dest_ctg_dir,frame_name)
                cv2.imwrite(frame_dest_dir, res_concat_img)
            