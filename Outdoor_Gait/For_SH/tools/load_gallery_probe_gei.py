
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def load_testing_data_gei(src_dir, condition):
    
    
    gallery_urls=[]
    probe_urls=[]
    
    gallery_images=[]
    gallery_infos=[]
    
    probe_images=[]
    probe_infos=[]
    
    id = ["%03d" % i for i in range(70, 139)]

# ##### nm-nm

    if(condition=='nm_nm'):

        scene1_list=[]
        scene2_list=[]
        scene3_list=[] 
        
        for i in range(len(id)):
            for l in os.listdir(os.path.join(src_dir, id[i])):
                scene = l.split("_")[0]
                condition = l.split("_")[1]
                if(condition == 'nm' and scene == 'scene1'):
                    scene1_list.append(l)
                if (condition == 'nm' and scene == 'scene2'):
                    scene2_list.append(l)
                if (condition == 'nm' and scene == 'scene3'):
                    scene3_list.append(l)
                    
############################# Galleries

            
            path=os.path.join(src_dir, id[i],scene1_list[0])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene1_list[1])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene2_list[0])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene2_list[1])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene3_list[0])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene3_list[1])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            
#######################Probes

            
            path=os.path.join(src_dir, id[i],scene1_list[2])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            probe_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            probe_infos.append(label)
            
            if(scene1_list[3]):
            
                path=os.path.join(src_dir, id[i],scene1_list[3])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene2_list[2])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            probe_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            probe_infos.append(label)
            
            if(scene2_list[3]):
                
                path=os.path.join(src_dir, id[i],scene2_list[3])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene3_list[2])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            probe_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            probe_infos.append(label)
            
            if(scene3_list[3]):
                
                path=os.path.join(src_dir, id[i],scene3_list[3])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            
            scene1_list=[]
            scene2_list=[]
            scene3_list=[] 


#################### nm-bg

    if(condition=='nm_bg'):

        scene1_list=[]
        scene2_list=[]
        scene3_list=[] 
        
        scene1_bg_list=[]
        scene2_bg_list=[]
        scene3_bg_list=[] 
        
        for i in range(len(id)):
            for l in os.listdir(os.path.join(src_dir, id[i])):
                scene = l.split("_")[0]
                condition = l.split("_")[1]
                if(condition == 'nm' and scene == 'scene1'):
                    scene1_list.append(l)
                if (condition == 'nm' and scene == 'scene2'):
                    scene2_list.append(l)
                if (condition == 'nm' and scene == 'scene3'):
                    scene3_list.append(l)
                    
                if(condition == 'bg' and scene == 'scene1'):
                    scene1_bg_list.append(l)
                if (condition == 'bg' and scene == 'scene2'):
                    scene2_bg_list.append(l)
                if (condition == 'bg' and scene == 'scene3'):
                    scene3_bg_list.append(l)
                    
################################ Galleries

            
            for j in range(len(scene1_list)):
                path=os.path.join(src_dir, id[i],scene1_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)

            for j in range(len(scene2_list)):
                path=os.path.join(src_dir, id[i],scene2_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)
            
            for j in range(len(scene3_list)):
                path=os.path.join(src_dir, id[i],scene3_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)
                
                
#############################Probes

            for j in range(len(scene1_bg_list)):
                path=os.path.join(src_dir, id[i],scene1_bg_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            for j in range(len(scene2_bg_list)):
                path=os.path.join(src_dir, id[i],scene2_bg_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
                
            for j in range(len(scene3_bg_list)):
            
                path=os.path.join(src_dir, id[i],scene3_bg_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)

            
            scene1_list=[]
            scene2_list=[]
            scene3_list=[] 
            
            scene1_bg_list=[]
            scene2_bg_list=[]
            scene3_bg_list=[] 


########### nm-cl

    if(condition=='nm_cl'):

        scene1_list=[]
        scene2_list=[]
        scene3_list=[] 
        
        scene1_cl_list=[]
        scene2_cl_list=[]
        scene3_cl_list=[] 
        
        for i in range(len(id)):
            for l in os.listdir(os.path.join(src_dir, id[i])):
                scene = l.split("_")[0]
                condition = l.split("_")[1]
                if(condition == 'nm' and scene == 'scene1'):
                    scene1_list.append(l)
                if (condition == 'nm' and scene == 'scene2'):
                    scene2_list.append(l)
                if (condition == 'nm' and scene == 'scene3'):
                    scene3_list.append(l)
                
                if(condition == 'cl' and scene == 'scene1'):
                    scene1_cl_list.append(l)
                if (condition == 'cl' and scene == 'scene2'):
                    scene2_cl_list.append(l)
                if (condition == 'cl' and scene == 'scene3'):
                    scene3_cl_list.append(l)
                    
                    
############################### Galleries

            
            for j in range(len(scene1_list)):
                path=os.path.join(src_dir, id[i],scene1_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)
                
            for j in range(len(scene2_list)):
                path=os.path.join(src_dir, id[i],scene2_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)
            
            for j in range(len(scene3_list)):
                path=os.path.join(src_dir, id[i],scene3_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)
            
            
################################Probes

            for j in range(len(scene1_cl_list)):
                path=os.path.join(src_dir, id[i],scene1_cl_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            for j in range(len(scene2_cl_list)):
                path=os.path.join(src_dir, id[i],scene2_cl_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)

            for j in range(len(scene3_cl_list)):
                path=os.path.join(src_dir, id[i],scene3_cl_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            scene1_list=[]
            scene2_list=[]
            scene3_list=[] 
            
            scene1_cl_list=[]
            scene2_cl_list=[]
            scene3_cl_list=[] 




#################### cl-nm

    if(condition=='cl_nm'):

        scene1_cl_list=[]
        scene2_cl_list=[]
        scene3_cl_list=[] 
        
        scene1_list=[]
        scene2_list=[]
        scene3_list=[] 
        
        for i in range(len(id)):
            for l in os.listdir(os.path.join(src_dir, id[i])):
                scene = l.split("_")[0]
                condition = l.split("_")[1]
                if(condition == 'nm' and scene == 'scene1'):
                    scene1_list.append(l)
                if (condition == 'nm' and scene == 'scene2'):
                    scene2_list.append(l)
                if (condition == 'nm' and scene == 'scene3'):
                    scene3_list.append(l)
                    
                if(condition == 'cl' and scene == 'scene1'):
                    scene1_cl_list.append(l)
                if (condition == 'cl' and scene == 'scene2'):
                    scene2_cl_list.append(l)
                if (condition == 'cl' and scene == 'scene3'):
                    scene3_cl_list.append(l)
                    
#galleries
            for j in range(len(scene1_cl_list)):
                path=os.path.join(src_dir, id[i],scene1_cl_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)

            for j in range(len(scene2_cl_list)):
                path=os.path.join(src_dir, id[i],scene2_cl_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)
            
            for j in range(len(scene3_cl_list)):
                path=os.path.join(src_dir, id[i],scene3_cl_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)

#probes
            
            for j in range(len(scene1_list)):
                path=os.path.join(src_dir, id[i],scene1_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            for j in range(len(scene2_list)):
                path=os.path.join(src_dir, id[i],scene2_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
                
            for j in range(len(scene3_list)):
            
                path=os.path.join(src_dir, id[i],scene3_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)

            
            scene1_list=[]
            scene2_list=[]
            scene3_list=[] 
            
            scene1_cl_list=[]
            scene2_cl_list=[]
            scene3_cl_list=[] 


##cl-cl

    if(condition=='cl_cl'):

        scene1_list=[]
        scene2_list=[]
        scene3_list=[] 
        
        for i in range(len(id)):
            for l in os.listdir(os.path.join(src_dir, id[i])):
                scene = l.split("_")[0]
                condition = l.split("_")[1]
                if(condition == 'cl' and scene == 'scene1'):
                    scene1_list.append(l)
                if (condition == 'cl' and scene == 'scene2'):
                    scene2_list.append(l)
                if (condition == 'cl' and scene == 'scene3'):
                    scene3_list.append(l)
                    
############################# Galleries

            
            path=os.path.join(src_dir, id[i],scene1_list[0])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene1_list[1])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene2_list[0])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene2_list[1])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene3_list[0])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene3_list[1])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            
#######################Probes

            
            path=os.path.join(src_dir, id[i],scene1_list[2])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            probe_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            probe_infos.append(label)
            
            if(scene1_list[3]):
            
                path=os.path.join(src_dir, id[i],scene1_list[3])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene2_list[2])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            probe_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            probe_infos.append(label)
            
            if(scene2_list[3]):
                
                path=os.path.join(src_dir, id[i],scene2_list[3])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene3_list[2])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            probe_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            probe_infos.append(label)
            
            if(scene3_list[3]):
                
                path=os.path.join(src_dir, id[i],scene3_list[3])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            
            scene1_list=[]
            scene2_list=[]
            scene3_list=[] 



#################### cl-bg

    if(condition=='cl_bg'):

        scene1_cl_list=[]
        scene2_cl_list=[]
        scene3_cl_list=[] 
        
        scene1_bg_list=[]
        scene2_bg_list=[]
        scene3_bg_list=[] 
        
        for i in range(len(id)):
            for l in os.listdir(os.path.join(src_dir, id[i])):
                scene = l.split("_")[0]
                condition = l.split("_")[1]
                if(condition == 'bg' and scene == 'scene1'):
                    scene1_bg_list.append(l)
                if (condition == 'bg' and scene == 'scene2'):
                    scene2_bg_list.append(l)
                if (condition == 'bg' and scene == 'scene3'):
                    scene3_bg_list.append(l)
                    
                if(condition == 'cl' and scene == 'scene1'):
                    scene1_cl_list.append(l)
                if (condition == 'cl' and scene == 'scene2'):
                    scene2_cl_list.append(l)
                if (condition == 'cl' and scene == 'scene3'):
                    scene3_cl_list.append(l)
                    
#galleries
            for j in range(len(scene1_cl_list)):
                path=os.path.join(src_dir, id[i],scene1_cl_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)

            for j in range(len(scene2_cl_list)):
                path=os.path.join(src_dir, id[i],scene2_cl_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)
            
            for j in range(len(scene3_cl_list)):
                path=os.path.join(src_dir, id[i],scene3_cl_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)

#probes
            
            for j in range(len(scene1_bg_list)):
                path=os.path.join(src_dir, id[i],scene1_bg_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            for j in range(len(scene2_bg_list)):
                path=os.path.join(src_dir, id[i],scene2_bg_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
                
            for j in range(len(scene3_bg_list)):
            
                path=os.path.join(src_dir, id[i],scene3_bg_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)

            
            scene1_bg_list=[]
            scene2_bg_list=[]
            scene3_bg_list=[] 
            
            scene1_cl_list=[]
            scene2_cl_list=[]
            scene3_cl_list=[] 


#################### bg-nm

    if(condition=='bg_nm'):

        scene1_list=[]
        scene2_list=[]
        scene3_list=[] 
        
        scene1_bg_list=[]
        scene2_bg_list=[]
        scene3_bg_list=[] 
        
        for i in range(len(id)):
            for l in os.listdir(os.path.join(src_dir, id[i])):
                scene = l.split("_")[0]
                condition = l.split("_")[1]
                if(condition == 'nm' and scene == 'scene1'):
                    scene1_list.append(l)
                if (condition == 'nm' and scene == 'scene2'):
                    scene2_list.append(l)
                if (condition == 'nm' and scene == 'scene3'):
                    scene3_list.append(l)
                    
                if(condition == 'bg' and scene == 'scene1'):
                    scene1_bg_list.append(l)
                if (condition == 'bg' and scene == 'scene2'):
                    scene2_bg_list.append(l)
                if (condition == 'bg' and scene == 'scene3'):
                    scene3_bg_list.append(l)
                    
################################ Galleries

            
            for j in range(len(scene1_bg_list)):
                path=os.path.join(src_dir, id[i],scene1_bg_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)

            for j in range(len(scene2_bg_list)):
                path=os.path.join(src_dir, id[i],scene2_bg_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)
            
            for j in range(len(scene3_bg_list)):
                path=os.path.join(src_dir, id[i],scene3_bg_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)
                
                
#############################Probes

            for j in range(len(scene1_list)):
                path=os.path.join(src_dir, id[i],scene1_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            for j in range(len(scene2_list)):
                path=os.path.join(src_dir, id[i],scene2_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
                
            for j in range(len(scene3_list)):
            
                path=os.path.join(src_dir, id[i],scene3_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)

            
            scene1_list=[]
            scene2_list=[]
            scene3_list=[] 
            
            scene1_bg_list=[]
            scene2_bg_list=[]
            scene3_bg_list=[] 


#################### bg-cl

    if(condition=='bg_cl'):

        scene1_cl_list=[]
        scene2_cl_list=[]
        scene3_cl_list=[] 
        
        scene1_bg_list=[]
        scene2_bg_list=[]
        scene3_bg_list=[] 
        
        for i in range(len(id)):
            for l in os.listdir(os.path.join(src_dir, id[i])):
                scene = l.split("_")[0]
                condition = l.split("_")[1]
                if(condition == 'bg' and scene == 'scene1'):
                    scene1_bg_list.append(l)
                if (condition == 'bg' and scene == 'scene2'):
                    scene2_bg_list.append(l)
                if (condition == 'bg' and scene == 'scene3'):
                    scene3_bg_list.append(l)
                    
                if(condition == 'cl' and scene == 'scene1'):
                    scene1_cl_list.append(l)
                if (condition == 'cl' and scene == 'scene2'):
                    scene2_cl_list.append(l)
                if (condition == 'cl' and scene == 'scene3'):
                    scene3_cl_list.append(l)
                    
#galleries
            for j in range(len(scene1_bg_list)):
                path=os.path.join(src_dir, id[i],scene1_bg_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)

            for j in range(len(scene2_bg_list)):
                path=os.path.join(src_dir, id[i],scene2_bg_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)
            
            for j in range(len(scene3_bg_list)):
                path=os.path.join(src_dir, id[i],scene3_bg_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                gallery_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                gallery_infos.append(label)

#probes
            
            for j in range(len(scene1_cl_list)):
                path=os.path.join(src_dir, id[i],scene1_cl_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            for j in range(len(scene2_cl_list)):
                path=os.path.join(src_dir, id[i],scene2_cl_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
                
            for j in range(len(scene3_cl_list)):
            
                path=os.path.join(src_dir, id[i],scene3_cl_list[j])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)

            
            scene1_bg_list=[]
            scene2_bg_list=[]
            scene3_bg_list=[] 
            
            scene1_cl_list=[]
            scene2_cl_list=[]
            scene3_cl_list=[] 

# ##### bg-bg

    if(condition=='bg_bg'):

        scene1_list=[]
        scene2_list=[]
        scene3_list=[] 
        
        for i in range(len(id)):
            for l in os.listdir(os.path.join(src_dir, id[i])):
                scene = l.split("_")[0]
                condition = l.split("_")[1]
                if(condition == 'bg' and scene == 'scene1'):
                    scene1_list.append(l)
                if (condition == 'bg' and scene == 'scene2'):
                    scene2_list.append(l)
                if (condition == 'bg' and scene == 'scene3'):
                    scene3_list.append(l)
                    
############################# Galleries

            
            path=os.path.join(src_dir, id[i],scene1_list[0])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene1_list[1])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene2_list[0])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene2_list[1])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene3_list[0])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene3_list[1])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            gallery_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            gallery_infos.append(label)
            
            
#######################Probes

            
            path=os.path.join(src_dir, id[i],scene1_list[2])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            probe_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            probe_infos.append(label)
            
            if(scene1_list[3]):
            
                path=os.path.join(src_dir, id[i],scene1_list[3])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene2_list[2])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            probe_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            probe_infos.append(label)
            
            if(scene2_list[3]):
                
                path=os.path.join(src_dir, id[i],scene2_list[3])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            path=os.path.join(src_dir, id[i],scene3_list[2])
            img = load_img(path, target_size=(224, 224,1))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:,:,0], axis=2)
            gray_img_3channel=x.repeat(3,axis=-1)
            probe_images.append(gray_img_3channel)
            label = "{0:03}".format(int(id[i].split("-")[0])-1)
            probe_infos.append(label)
            
            if(scene3_list[3]):
                
                path=os.path.join(src_dir, id[i],scene3_list[3])
                img = load_img(path, target_size=(224, 224,1))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                gray_img_3channel=x.repeat(3,axis=-1)
                probe_images.append(gray_img_3channel)
                label = "{0:03}".format(int(id[i].split("-")[0])-1)
                probe_infos.append(label)
            
            
            scene1_list=[]
            scene2_list=[]
            scene3_list=[] 


  
    return gallery_images, gallery_infos,probe_images,probe_infos



                
            
 



