
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical



#train
def load_training_data(src_dir):
    
    id = ["{0:03}".format(i) for i in range(1, 75)]
    categories = ["bg-01", "bg-02", "cl-01", "cl-02",
                  "nm-01", "nm-02", "nm-03", "nm-04",
                  "nm-05", "nm-06"]
    
    training_x = []
    train_label = []
    
    for i in range(len(id)):
        for j in range(len(categories)):
                for l in os.listdir(os.path.join(src_dir, id[i], categories[j])):
                    path=os.path.join(src_dir, id[i], categories[j], l)
                    img = load_img(path, target_size=(224, 224,1))
                    x3d = img_to_array(img)
                    x = np.expand_dims(x3d[:,:,0], axis=2)
                    gray_img_3channel=x.repeat(3,axis=-1)
                    training_x.append(gray_img_3channel)
                    label = "{0:03}".format(int(l.split("-")[0])-1)
                    train_label.append(label)

    training_y = to_categorical(train_label)
    training_x= np.array(training_x)
    training_y = np.array(training_y)
    
    return training_x, training_y



def load_testing_data(src_dir):
    
    gallery_images=[]
    gallery_infos=[]
    
    probe_images=[]
    probe_infos=[]
    
    id = ["%03d" % i for i in range(75, 125)]
    
#gallery

    categories = ["nm-01", "nm-02", "nm-03", "nm-04"]
    
    for i in range(len(id)):
        for j in range(len(categories)):
                for l in os.listdir(os.path.join(src_dir, id[i], categories[j])):
                    path=os.path.join(src_dir, id[i], categories[j], l)
                    img = load_img(path, target_size=(224, 224,1))
                    x3d = img_to_array(img)
                    x = np.expand_dims(x3d[:,:,0], axis=2)
                    gray_img_3channel=x.repeat(3,axis=-1)
                    gallery_images.append(gray_img_3channel)
                    
                    label = "{0:03}".format(int(l.split("-")[0])-1)
                    aci="{0:03}".format(int(l.split("-")[3].split(".")[0]))
                    gallery_infos.append((label, aci))
                
    
    

#probe

    categories = ["nm-05", "nm-06"]
    # categories = ["bg-01", "bg-02"]
    # categories = ["cl-01", "cl-02"]
    
    
    for i in range(len(id)):
        for j in range(len(categories)):
                for l in os.listdir(os.path.join(src_dir, id[i], categories[j])):
                    path=os.path.join(src_dir, id[i], categories[j], l)
                    img = load_img(path, target_size=(224, 224,1))
                    x3d = img_to_array(img)
                    x = np.expand_dims(x3d[:,:,0], axis=2)
                    gray_img_3channel=x.repeat(3,axis=-1)
                    probe_images.append(gray_img_3channel)
                    
                    label = "{0:03}".format(int(l.split("-")[0])-1)
                    aci="{0:03}".format(int(l.split("-")[3].split(".")[0]))
                    probe_infos.append((label, aci))
                    
    
    return gallery_images, gallery_infos,probe_images,probe_infos





















