
import os
import cv2
import glob
import numpy as np
from time import sleep
from skimage.io import imread, imshow

T_H = 224
T_W = 224

Path_silhoulettes = 'data/GaitDatasetB-silh'
Path_OF_Frames = 'data/Optical_Flow_Frames'
Path_Dest= 'data/OFs'


id = ["{0:03}".format(i) for i in range(1,125)]

categories = ["bg-01", "bg-02", "cl-01", "cl-02",
                "nm-01", "nm-02", "nm-03", "nm-04",
                "nm-05", "nm-06"]

angles = ["{0:03}".format(i) for i in range(0, 181, 18)]

for m in range(len(id)):
    if id[m] =='005':
        continue
    for n in range(len(categories)):
        for k in range(len(angles)):
            print('committed:{}/{}/{}',id[m],categories[n],angles[k])
            print('/n')
            # for l in os.listdir(os.path.join(Path_silhoulettes, id[m], categories[n],angles[k])):
            frames = sorted(glob.glob(os.path.join(Path_silhoulettes, id[m], categories[n],angles[k],'*')))
            if len(frames) == 0:
                continue
            for ind, img_path in enumerate(frames[:-1]):
                
                path_img = frames[ind+1]
                path_img2 = os.path.join(Path_OF_Frames, id[m], categories[n], angles[k], os.path.basename(frames[ind+1]))
                dest_path = os.path.join(Path_Dest, id[m], categories[n],angles[k],os.path.basename(frames[ind+1]))

                img = cv2.imread(path_img)[:, :, 0]
                img2 = cv2.imread(path_img2)
                
                
                if cv2.imread(frames[ind])[:, :, 0].sum() <= 10000:
                    continue
                if img.sum() <= 10000:
                    continue
                
                # Get the top and bottom point
                y = img.sum(axis=1)
                y_top = (y != 0).argmax(axis=0)
                y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
                img = img[y_top:y_btm + 1, :]
                img2 = img2[y_top-10:y_btm + 10, :,:]
                if img2.shape[0]<=20:
                    img2 = cv2.imread(path_img2)
                    img2 = img2[y_top:y_btm + 1, :,:] 
                # As the height of a person is larger than the width,
                # use the height to calculate resize ratio.
                _r = img.shape[1] / img.shape[0]
                _t_w = int(T_H * _r)
                img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
                img2 = cv2.resize(img2, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)

                # Get the median of x axis and regard it as the x center of the person.
                sum_point = img.sum()
                sum_column = img.sum(axis=0).cumsum()
                x_center = -1
                for i in range(sum_column.size):
                    if sum_column[i] > sum_point / 2:
                        x_center = i
                        break
                if x_center < 0:
                    print('frame no center.')

                h_T_W = int(T_W / 2)
                left = x_center - h_T_W
                right = x_center + h_T_W
                if left <= 0 or right >= img.shape[1]:
                    left += h_T_W
                    right += h_T_W
                    _ = np.zeros((img.shape[0], h_T_W))
                    img = np.concatenate([_, img, _], axis=1)
                    
                    (B, G, R) = cv2.split(img2)
                    B=np.concatenate([_, B, _], axis=1)
                    G=np.concatenate([_, G, _], axis=1)
                    R=np.concatenate([_, R, _], axis=1)
                    merged = cv2.merge([B, G, R])
                    img2 = merged
                    
                img = img[:, left:right]
                img2 = img2[:, left:right,:]
                result_img = img.astype('uint8')
                result_img2 = img2.astype('uint8')

                cv2.imwrite(dest_path,result_img2)
                
                
                
