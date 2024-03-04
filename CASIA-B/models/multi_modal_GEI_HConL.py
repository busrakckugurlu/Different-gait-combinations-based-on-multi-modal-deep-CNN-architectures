
import numpy as np
import random
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.backend import set_session
from keras.models import Model
from keras.models import load_model
from keras.layers import Concatenate
from tools.create_train_test_data import load_testing_data


gallery_images_gei, gallery_infos_gei, probe_images_gei, probe_infos_gei = load_testing_data("data/Casia_gait/gei")
gallery_images_hconl, gallery_infos_hconl, probe_images_hconl, probe_infos_hconl = load_testing_data('data/HConL')



######################################################################

def extract_feature(images_list1, images_list2,net):
    
  features = []
  for img1,img2 in zip(images_list1,images_list2):
    hconl_x= np.expand_dims(img1, axis=0)
    gei_x= np.expand_dims(img2, axis=0)
    
    two_stream_x = [hconl_x, gei_x]
    
    feature = net.predict(two_stream_x)
    features.append(np.squeeze(feature))
  return features


######################################################### GALLERIES GEI
gallery_000_images_gei=[]
gallery_000_infos_gei=[]

gallery_018_images_gei=[]
gallery_018_infos_gei=[]

gallery_036_images_gei=[]
gallery_036_infos_gei=[]

gallery_054_images_gei=[]
gallery_054_infos_gei=[]

gallery_072_images_gei=[]
gallery_072_infos_gei=[]

gallery_090_images_gei=[]
gallery_090_infos_gei=[]

gallery_108_images_gei=[]
gallery_108_infos_gei=[]

gallery_126_images_gei=[]
gallery_126_infos_gei=[]

gallery_144_images_gei=[]
gallery_144_infos_gei=[]

gallery_162_images_gei=[]
gallery_162_infos_gei=[]

gallery_180_images_gei=[]
gallery_180_infos_gei=[]

for gim_gei, gin_gei in zip(gallery_images_gei,gallery_infos_gei):
    if(gin_gei[1]=='000'):
        gallery_000_images_gei.append(gim_gei)
        gallery_000_infos_gei.append(gin_gei)
    if(gin_gei[1]=='018'):
        gallery_018_images_gei.append(gim_gei)
        gallery_018_infos_gei.append(gin_gei)
    if(gin_gei[1]=='036'):
        gallery_036_images_gei.append(gim_gei)
        gallery_036_infos_gei.append(gin_gei)
    if(gin_gei[1]=='054'):
        gallery_054_images_gei.append(gim_gei)
        gallery_054_infos_gei.append(gin_gei)
    if(gin_gei[1]=='072'):
        gallery_072_images_gei.append(gim_gei)
        gallery_072_infos_gei.append(gin_gei)
    if(gin_gei[1]=='090'):
        gallery_090_images_gei.append(gim_gei)
        gallery_090_infos_gei.append(gin_gei)
    if(gin_gei[1]=='108'):
        gallery_108_images_gei.append(gim_gei)
        gallery_108_infos_gei.append(gin_gei)
    if(gin_gei[1]=='126'):
        gallery_126_images_gei.append(gim_gei)
        gallery_126_infos_gei.append(gin_gei)
    if(gin_gei[1]=='144'):
        gallery_144_images_gei.append(gim_gei)
        gallery_144_infos_gei.append(gin_gei)
    if(gin_gei[1]=='162'):
        gallery_162_images_gei.append(gim_gei)
        gallery_162_infos_gei.append(gin_gei)
    if(gin_gei[1]=='180'):
        gallery_180_images_gei.append(gim_gei)
        gallery_180_infos_gei.append(gin_gei)



################################################################### PROBES GEI
probe_000_images_gei=[]
probe_000_infos_gei=[]

probe_018_images_gei=[]
probe_018_infos_gei=[]

probe_036_images_gei=[]
probe_036_infos_gei=[]

probe_054_images_gei=[]
probe_054_infos_gei=[]

probe_072_images_gei=[]
probe_072_infos_gei=[]

probe_090_images_gei=[]
probe_090_infos_gei=[]

probe_108_images_gei=[]
probe_108_infos_gei=[]

probe_126_images_gei=[]
probe_126_infos_gei=[]

probe_144_images_gei=[]
probe_144_infos_gei=[]

probe_162_images_gei=[]
probe_162_infos_gei=[]

probe_180_images_gei=[]
probe_180_infos_gei=[]

for pim_gei, pin_gei in zip(probe_images_gei,probe_infos_gei):
    if(pin_gei[1]=='000'):
        probe_000_images_gei.append(pim_gei)
        probe_000_infos_gei.append(pin_gei)
    if(pin_gei[1]=='018'):
        probe_018_images_gei.append(pim_gei)
        probe_018_infos_gei.append(pin_gei)
    if(pin_gei[1]=='036'):
        probe_036_images_gei.append(pim_gei)
        probe_036_infos_gei.append(pin_gei)
    if(pin_gei[1]=='054'):
        probe_054_images_gei.append(pim_gei)
        probe_054_infos_gei.append(pin_gei)
    if(pin_gei[1]=='072'):
        probe_072_images_gei.append(pim_gei)
        probe_072_infos_gei.append(pin_gei)
    if(pin_gei[1]=='090'):
        probe_090_images_gei.append(pim_gei)
        probe_090_infos_gei.append(pin_gei)
    if(pin_gei[1]=='108'):
        probe_108_images_gei.append(pim_gei)
        probe_108_infos_gei.append(pin_gei)
    if(pin_gei[1]=='126'):
        probe_126_images_gei.append(pim_gei)
        probe_126_infos_gei.append(pin_gei)
    if(pin_gei[1]=='144'):
        probe_144_images_gei.append(pim_gei)
        probe_144_infos_gei.append(pin_gei)
    if(pin_gei[1]=='162'):
        probe_162_images_gei.append(pim_gei)
        probe_162_infos_gei.append(pin_gei)
    if(pin_gei[1]=='180'):
        probe_180_images_gei.append(pim_gei)
        probe_180_infos_gei.append(pin_gei)
        

######################################################################## ALL GALERIES GEI

All_gallery_angles_images_gei=[]
All_gallery_angles_infos_gei=[]

All_gallery_angles_images_gei.append(gallery_000_images_gei)
All_gallery_angles_infos_gei.append(gallery_000_infos_gei)

All_gallery_angles_images_gei.append(gallery_018_images_gei)
All_gallery_angles_infos_gei.append(gallery_018_infos_gei)

All_gallery_angles_images_gei.append(gallery_036_images_gei)
All_gallery_angles_infos_gei.append(gallery_036_infos_gei)

All_gallery_angles_images_gei.append(gallery_054_images_gei)
All_gallery_angles_infos_gei.append(gallery_054_infos_gei)

All_gallery_angles_images_gei.append(gallery_072_images_gei)
All_gallery_angles_infos_gei.append(gallery_072_infos_gei)

All_gallery_angles_images_gei.append(gallery_090_images_gei)
All_gallery_angles_infos_gei.append(gallery_090_infos_gei)

All_gallery_angles_images_gei.append(gallery_108_images_gei)
All_gallery_angles_infos_gei.append(gallery_108_infos_gei)

All_gallery_angles_images_gei.append(gallery_126_images_gei)
All_gallery_angles_infos_gei.append(gallery_126_infos_gei)

All_gallery_angles_images_gei.append(gallery_144_images_gei)
All_gallery_angles_infos_gei.append(gallery_144_infos_gei)

All_gallery_angles_images_gei.append(gallery_162_images_gei)
All_gallery_angles_infos_gei.append(gallery_162_infos_gei)

All_gallery_angles_images_gei.append(gallery_180_images_gei)
All_gallery_angles_infos_gei.append(gallery_180_infos_gei)
########################################################################### ALL PROBES GEI

All_probe_angles_images_gei=[]
All_probe_angles_infos_gei=[]

All_probe_angles_images_gei.append(probe_000_images_gei)
All_probe_angles_infos_gei.append(probe_000_infos_gei)

All_probe_angles_images_gei.append(probe_018_images_gei)
All_probe_angles_infos_gei.append(probe_018_infos_gei)

All_probe_angles_images_gei.append(probe_036_images_gei)
All_probe_angles_infos_gei.append(probe_036_infos_gei)

All_probe_angles_images_gei.append(probe_054_images_gei)
All_probe_angles_infos_gei.append(probe_054_infos_gei)

All_probe_angles_images_gei.append(probe_072_images_gei)
All_probe_angles_infos_gei.append(probe_072_infos_gei)

All_probe_angles_images_gei.append(probe_090_images_gei)
All_probe_angles_infos_gei.append(probe_090_infos_gei)

All_probe_angles_images_gei.append(probe_108_images_gei)
All_probe_angles_infos_gei.append(probe_108_infos_gei)

All_probe_angles_images_gei.append(probe_126_images_gei)
All_probe_angles_infos_gei.append(probe_126_infos_gei)

All_probe_angles_images_gei.append(probe_144_images_gei)
All_probe_angles_infos_gei.append(probe_144_infos_gei)

All_probe_angles_images_gei.append(probe_162_images_gei)
All_probe_angles_infos_gei.append(probe_162_infos_gei)

All_probe_angles_images_gei.append(probe_180_images_gei)
All_probe_angles_infos_gei.append(probe_180_infos_gei)


######################################################### GALLERIES STACK
gallery_000_images_hconl=[]
gallery_000_infos_hconl=[]

gallery_018_images_hconl=[]
gallery_018_infos_hconl=[]

gallery_036_images_hconl=[]
gallery_036_infos_hconl=[]

gallery_054_images_hconl=[]
gallery_054_infos_hconl=[]

gallery_072_images_hconl=[]
gallery_072_infos_hconl=[]

gallery_090_images_hconl=[]
gallery_090_infos_hconl=[]

gallery_108_images_hconl=[]
gallery_108_infos_hconl=[]

gallery_126_images_hconl=[]
gallery_126_infos_hconl=[]

gallery_144_images_hconl=[]
gallery_144_infos_hconl=[]

gallery_162_images_hconl=[]
gallery_162_infos_hconl=[]

gallery_180_images_hconl=[]
gallery_180_infos_hconl=[]

for gim_hconl, gin_hconl in zip(gallery_images_hconl,gallery_infos_hconl):
    if(gin_hconl[1]=='000'):
        gallery_000_images_hconl.append(gim_hconl)
        gallery_000_infos_hconl.append(gin_hconl)
    if(gin_hconl[1]=='018'):
        gallery_018_images_hconl.append(gim_hconl)
        gallery_018_infos_hconl.append(gin_hconl)
    if(gin_hconl[1]=='036'):
        gallery_036_images_hconl.append(gim_hconl)
        gallery_036_infos_hconl.append(gin_hconl)
    if(gin_hconl[1]=='054'):
        gallery_054_images_hconl.append(gim_hconl)
        gallery_054_infos_hconl.append(gin_hconl)
    if(gin_hconl[1]=='072'):
        gallery_072_images_hconl.append(gim_hconl)
        gallery_072_infos_hconl.append(gin_hconl)
    if(gin_hconl[1]=='090'):
        gallery_090_images_hconl.append(gim_hconl)
        gallery_090_infos_hconl.append(gin_hconl)
    if(gin_hconl[1]=='108'):
        gallery_108_images_hconl.append(gim_hconl)
        gallery_108_infos_hconl.append(gin_hconl)
    if(gin_hconl[1]=='126'):
        gallery_126_images_hconl.append(gim_hconl)
        gallery_126_infos_hconl.append(gin_hconl)
    if(gin_hconl[1]=='144'):
        gallery_144_images_hconl.append(gim_hconl)
        gallery_144_infos_hconl.append(gin_hconl)
    if(gin_hconl[1]=='162'):
        gallery_162_images_hconl.append(gim_hconl)
        gallery_162_infos_hconl.append(gin_hconl)
    if(gin_hconl[1]=='180'):
        gallery_180_images_hconl.append(gim_hconl)
        gallery_180_infos_hconl.append(gin_hconl)

################################################################### PROBES STACK
probe_000_images_hconl=[]
probe_000_infos_hconl=[]

probe_018_images_hconl=[]
probe_018_infos_hconl=[]

probe_036_images_hconl=[]
probe_036_infos_hconl=[]

probe_054_images_hconl=[]
probe_054_infos_hconl=[]

probe_072_images_hconl=[]
probe_072_infos_hconl=[]

probe_090_images_hconl=[]
probe_090_infos_hconl=[]

probe_108_images_hconl=[]
probe_108_infos_hconl=[]

probe_126_images_hconl=[]
probe_126_infos_hconl=[]

probe_144_images_hconl=[]
probe_144_infos_hconl=[]

probe_162_images_hconl=[]
probe_162_infos_hconl=[]

probe_180_images_hconl=[]
probe_180_infos_hconl=[]

for pim_hconl, pin_hconl in zip(probe_images_hconl,probe_infos_hconl):
    if(pin_hconl[1]=='000'):
        probe_000_images_hconl.append(pim_hconl)
        probe_000_infos_hconl.append(pin_hconl)
    if(pin_hconl[1]=='018'):
        probe_018_images_hconl.append(pim_hconl)
        probe_018_infos_hconl.append(pin_hconl)
    if(pin_hconl[1]=='036'):
        probe_036_images_hconl.append(pim_hconl)
        probe_036_infos_hconl.append(pin_hconl)
    if(pin_hconl[1]=='054'):
        probe_054_images_hconl.append(pim_hconl)
        probe_054_infos_hconl.append(pin_hconl)
    if(pin_hconl[1]=='072'):
        probe_072_images_hconl.append(pim_hconl)
        probe_072_infos_hconl.append(pin_hconl)
    if(pin_hconl[1]=='090'):
        probe_090_images_hconl.append(pim_hconl)
        probe_090_infos_hconl.append(pin_hconl)
    if(pin_hconl[1]=='108'):
        probe_108_images_hconl.append(pim_hconl)
        probe_108_infos_hconl.append(pin_hconl)
    if(pin_hconl[1]=='126'):
        probe_126_images_hconl.append(pim_hconl)
        probe_126_infos_hconl.append(pin_hconl)
    if(pin_hconl[1]=='144'):
        probe_144_images_hconl.append(pim_hconl)
        probe_144_infos_hconl.append(pin_hconl)
    if(pin_hconl[1]=='162'):
        probe_162_images_hconl.append(pim_hconl)
        probe_162_infos_hconl.append(pin_hconl)
    if(pin_hconl[1]=='180'):
        probe_180_images_hconl.append(pim_hconl)
        probe_180_infos_hconl.append(pin_hconl)
        

######################################################################## ALL GALERIES STACK

All_gallery_angles_images_hconl=[]
All_gallery_angles_infos_hconl=[]

All_gallery_angles_images_hconl.append(gallery_000_images_hconl)
All_gallery_angles_infos_hconl.append(gallery_000_infos_hconl)

All_gallery_angles_images_hconl.append(gallery_018_images_hconl)
All_gallery_angles_infos_hconl.append(gallery_018_infos_hconl)

All_gallery_angles_images_hconl.append(gallery_036_images_hconl)
All_gallery_angles_infos_hconl.append(gallery_036_infos_hconl)

All_gallery_angles_images_hconl.append(gallery_054_images_hconl)
All_gallery_angles_infos_hconl.append(gallery_054_infos_hconl)

All_gallery_angles_images_hconl.append(gallery_072_images_hconl)
All_gallery_angles_infos_hconl.append(gallery_072_infos_hconl)

All_gallery_angles_images_hconl.append(gallery_090_images_hconl)
All_gallery_angles_infos_hconl.append(gallery_090_infos_hconl)

All_gallery_angles_images_hconl.append(gallery_108_images_hconl)
All_gallery_angles_infos_hconl.append(gallery_108_infos_hconl)

All_gallery_angles_images_hconl.append(gallery_126_images_hconl)
All_gallery_angles_infos_hconl.append(gallery_126_infos_hconl)

All_gallery_angles_images_hconl.append(gallery_144_images_hconl)
All_gallery_angles_infos_hconl.append(gallery_144_infos_hconl)

All_gallery_angles_images_hconl.append(gallery_162_images_hconl)
All_gallery_angles_infos_hconl.append(gallery_162_infos_hconl)

All_gallery_angles_images_hconl.append(gallery_180_images_hconl)
All_gallery_angles_infos_hconl.append(gallery_180_infos_hconl)
########################################################################### ALL PROBES STACK

All_probe_angles_images_hconl=[]
All_probe_angles_infos_hconl=[]

All_probe_angles_images_hconl.append(probe_000_images_hconl)
All_probe_angles_infos_hconl.append(probe_000_infos_hconl)

All_probe_angles_images_hconl.append(probe_018_images_hconl)
All_probe_angles_infos_hconl.append(probe_018_infos_hconl)

All_probe_angles_images_hconl.append(probe_036_images_hconl)
All_probe_angles_infos_hconl.append(probe_036_infos_hconl)

All_probe_angles_images_hconl.append(probe_054_images_hconl)
All_probe_angles_infos_hconl.append(probe_054_infos_hconl)

All_probe_angles_images_hconl.append(probe_072_images_hconl)
All_probe_angles_infos_hconl.append(probe_072_infos_hconl)

All_probe_angles_images_hconl.append(probe_090_images_hconl)
All_probe_angles_infos_hconl.append(probe_090_infos_hconl)

All_probe_angles_images_hconl.append(probe_108_images_hconl)
All_probe_angles_infos_hconl.append(probe_108_infos_hconl)

All_probe_angles_images_hconl.append(probe_126_images_hconl)
All_probe_angles_infos_hconl.append(probe_126_infos_hconl)

All_probe_angles_images_hconl.append(probe_144_images_hconl)
All_probe_angles_infos_hconl.append(probe_144_infos_hconl)

All_probe_angles_images_hconl.append(probe_162_images_hconl)
All_probe_angles_infos_hconl.append(probe_162_infos_hconl)

All_probe_angles_images_hconl.append(probe_180_images_hconl)
All_probe_angles_infos_hconl.append(probe_180_infos_hconl)




####################### use GPU to calculate the similarity matrix
tf.disable_v2_behavior()
query_t = tf.placeholder(tf.float32, (None, None))
test_t = tf.placeholder(tf.float32, (None, None))
query_t_norm = tf.nn.l2_normalize(query_t, dim=1)
test_t_norm = tf.nn.l2_normalize(test_t, dim=1)
tensor = tf.matmul(query_t_norm, test_t_norm, transpose_a=False, transpose_b=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


            

import numpy as np
result_table=np.zeros((11, 11))


for i in range(len(All_probe_angles_images_gei)):
    for j in range(len(All_gallery_angles_images_gei)):
        
        g = tf.Graph()
        with g.as_default():
            
            hconl_modal = load_model('weights/MobileNet_HConL.ckpt')
            gei_modal = load_model('weights/MobileNet_GEI.ckpt')

            for layer in hconl_modal.layers:
                layer.trainable = False
                layer._name = 'hconl_' + layer._name
            for layer2 in gei_modal.layers:
                layer2.trainable = False
                layer2._name = 'gei_' + layer2._name

            hconl_output = hconl_modal.get_layer('hconl_global_average_pooling2d').output
            gei_output = gei_modal.get_layer('gei_global_average_pooling2d').output
            fused_output = Concatenate(name='fusion_layer')([hconl_output, gei_output])
            net = Model(inputs=[hconl_modal.input, gei_modal.input], outputs=fused_output, name='two_stream')
            
            test_features = extract_feature(All_gallery_angles_images_hconl[j], All_gallery_angles_images_gei[j], net)
            query_features = extract_feature(All_probe_angles_images_hconl[i], All_probe_angles_images_gei[i], net)

        result = sess.run(tensor, {query_t: query_features, test_t: test_features})
        
        Probe_num=len(All_probe_angles_images_gei[i])
        tmp_probe_infos=[]
        for k in range(len(All_probe_angles_images_gei[i])):
            tmp_probe_infos.append(0)
            
        indexes=[]
        for m in range(result.shape[0]):
            index=np.argmax(result[m])
            tmp_probe_infos[m]= All_gallery_angles_infos_gei[j][index] 
            indexes.append(index)
            
        true_count=0
        for n in range(len(tmp_probe_infos)):
            if tmp_probe_infos[n][0]==All_probe_angles_infos_gei[i][n][0]: 
                true_count=true_count+1
                
        
        success=100*true_count/Probe_num
        result_table[i][j]=success
        success=0
        
