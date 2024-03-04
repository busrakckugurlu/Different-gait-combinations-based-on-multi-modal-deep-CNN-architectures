
import glob
import os
import cv2
import numpy as np


Path_RGBframes = 'data/Casia-B-RGBframes'
Path_Dest= 'data/Optical_Flow_Frames'

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
            
            frames = sorted(glob.glob(os.path.join(Path_RGBframes, id[m], categories[n],angles[k],'*')))
            if len(frames) == 0:
                continue
            prvs = cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2GRAY)
            
            hsv_mask = np.zeros_like(cv2.imread(frames[0]))
            hsv_mask[..., 1] = 255
            
            for ind, img_path in enumerate(frames[:-1]):
                next = cv2.cvtColor(cv2.imread(frames[ind+1]), cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv_mask[..., 0] = ang * 180 / np.pi / 2
                hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
                # cv2.imshow('frame2', rgb_representation)
                kk = cv2.waitKey(20) & 0xff
                basename = os.path.splitext(os.path.basename(frames[ind+1]))[0]
                
                dest_path = os.path.join(Path_Dest, id[m], categories[n],angles[k],basename +'.png')
                cv2.imwrite(dest_path, rgb_representation)
                prvs = next
                
            cv2.destroyAllWindows()



               
                
               
                
               
                
               
                
               
                
               
                
               
                
               
                
               
                
               
                
               