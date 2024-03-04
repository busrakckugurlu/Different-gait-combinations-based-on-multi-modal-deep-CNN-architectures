import numpy as np
import os
from tensorflow.keras.utils import to_categorical


def load_testing_data_stack(src_dir):
    
    gallery_images=[]
    gallery_infos=[]
    
    probe_images=[]
    probe_infos=[]
    
    id = ["%03d" % i for i in range(75, 125)]
    angles = ["{0:03}".format(i) for i in range(0, 181, 18)]
    
#gallery

    categories = ["nm-01", "nm-02", "nm-03", "nm-04"]
    
    for i in range(len(id)):
        for j in range(len(categories)):
            for k in range(len(angles)):
                for l in os.listdir(os.path.join(src_dir, id[i], categories[j],angles[k])):
                    path=os.path.join(src_dir, id[i], categories[j],angles[k], l)
                    numpy_array = np.load(path)
                    gallery_images.append(numpy_array)
                    
                    label = "{0:03}".format(int(l.split("-")[0])-1)
                    aci="{0:03}".format(int(l.split("-")[3].split(".")[0]))
                    gallery_infos.append((label, aci))
                
    
    

#probe

    categories = ["nm-05", "nm-06"]
    # categories = ["bg-01", "bg-02"]
    # categories = ["cl-01", "cl-02"]
    
    
    for i in range(len(id)):
        for j in range(len(categories)):
            for k in range(len(angles)):
                for l in os.listdir(os.path.join(src_dir, id[i], categories[j],angles[k])):
                    path=os.path.join(src_dir, id[i], categories[j],angles[k], l)
                    numpy_array = np.load(path)
                    probe_images.append(numpy_array)
                    
                    label = "{0:03}".format(int(l.split("-")[0])-1)
                    aci="{0:03}".format(int(l.split("-")[3].split(".")[0]))
                    probe_infos.append((label, aci))
                    
    
    return gallery_images, gallery_infos,probe_images,probe_infos





















