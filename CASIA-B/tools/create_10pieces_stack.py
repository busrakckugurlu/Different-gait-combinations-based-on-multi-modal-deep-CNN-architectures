import os
import numpy as np
from numpy.random import randint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from collections import OrderedDict

src_dir= 'data/SHs'
dest_dir= 'data/10_pieces_SH'

#src_dir= 'data/OFs'
#dest_dir= 'data/10_pieces_OF'

dir_mapping = OrderedDict([(src_dir, dest_dir)])


count=0


                          
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
#angles    
            for angles in os.listdir(ctg_dir):
                ang_dir = os.path.join(ctg_dir, angles)
                dest_ang_dir = os.path.join(dest_dir, class_name,category,angles)
                if not os.path.exists(dest_ang_dir):
                    os.mkdir(dest_ang_dir)
#frames
                frame_sequence = []
                tmp_path_list=[]
                for img in os.listdir(ang_dir):
                    tmp_path=os.path.join(ang_dir,img)
                    tmp_path_list.append(tmp_path)
                if len(tmp_path_list)>10: 
                    indexes = randint(0, len(tmp_path_list), 10)
                    for index in indexes:                       
                        frame = load_img(tmp_path_list[index])
                        frame = img_to_array(frame)
                        frame_sequence.append(frame)
                    frame_sequence = np.stack(frame_sequence, axis=0)
                    frame_name=class_name+'-'+category+'-'+angles+'.npy'
                    frame_dest_dir = os.path.join(dest_ang_dir,frame_name)
                    np.save(frame_dest_dir, frame_sequence)
                    tmp_path_list.clear()
                else:
                    print(tmp_path_list)
                    print('\n')
                    tmp_path_list.clear()
                    continue
        









        