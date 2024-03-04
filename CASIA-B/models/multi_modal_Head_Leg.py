
import numpy as np
import random
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.backend import set_session
from keras.models import Model
from keras.models import load_model
from keras.layers import Concatenate
from tools.create_train_test_data import load_testing_data

gallery_images_head, gallery_infos_head, probe_images_head, probe_infos_head = load_testing_data('data/Cropped_Head')
gallery_images_leg, gallery_infos_leg, probe_images_leg, probe_infos_leg = load_testing_data('data/Cropped_Leg')



######################################################################

def extract_feature(images_list1, images_list2,net):
    
  features = []
  for img1,img2 in zip(images_list1,images_list2):
    leg_x= np.expand_dims(img1, axis=0)
    head_x= np.expand_dims(img2, axis=0)
    
    two_stream_x = [leg_x, head_x]
    
    feature = net.predict(two_stream_x)
    features.append(np.squeeze(feature))
  return features


######################################################### GALLERIES HEAD
gallery_000_images_head=[]
gallery_000_infos_head=[]

gallery_018_images_head=[]
gallery_018_infos_head=[]

gallery_036_images_head=[]
gallery_036_infos_head=[]

gallery_054_images_head=[]
gallery_054_infos_head=[]

gallery_072_images_head=[]
gallery_072_infos_head=[]

gallery_090_images_head=[]
gallery_090_infos_head=[]

gallery_108_images_head=[]
gallery_108_infos_head=[]

gallery_126_images_head=[]
gallery_126_infos_head=[]

gallery_144_images_head=[]
gallery_144_infos_head=[]

gallery_162_images_head=[]
gallery_162_infos_head=[]

gallery_180_images_head=[]
gallery_180_infos_head=[]

for gim_head, gin_head in zip(gallery_images_head,gallery_infos_head):
    if(gin_head[1]=='000'):
        gallery_000_images_head.append(gim_head)
        gallery_000_infos_head.append(gin_head)
    if(gin_head[1]=='018'):
        gallery_018_images_head.append(gim_head)
        gallery_018_infos_head.append(gin_head)
    if(gin_head[1]=='036'):
        gallery_036_images_head.append(gim_head)
        gallery_036_infos_head.append(gin_head)
    if(gin_head[1]=='054'):
        gallery_054_images_head.append(gim_head)
        gallery_054_infos_head.append(gin_head)
    if(gin_head[1]=='072'):
        gallery_072_images_head.append(gim_head)
        gallery_072_infos_head.append(gin_head)
    if(gin_head[1]=='090'):
        gallery_090_images_head.append(gim_head)
        gallery_090_infos_head.append(gin_head)
    if(gin_head[1]=='108'):
        gallery_108_images_head.append(gim_head)
        gallery_108_infos_head.append(gin_head)
    if(gin_head[1]=='126'):
        gallery_126_images_head.append(gim_head)
        gallery_126_infos_head.append(gin_head)
    if(gin_head[1]=='144'):
        gallery_144_images_head.append(gim_head)
        gallery_144_infos_head.append(gin_head)
    if(gin_head[1]=='162'):
        gallery_162_images_head.append(gim_head)
        gallery_162_infos_head.append(gin_head)
    if(gin_head[1]=='180'):
        gallery_180_images_head.append(gim_head)
        gallery_180_infos_head.append(gin_head)



################################################################### PROBES HEAD
probe_000_images_head=[]
probe_000_infos_head=[]

probe_018_images_head=[]
probe_018_infos_head=[]

probe_036_images_head=[]
probe_036_infos_head=[]

probe_054_images_head=[]
probe_054_infos_head=[]

probe_072_images_head=[]
probe_072_infos_head=[]

probe_090_images_head=[]
probe_090_infos_head=[]

probe_108_images_head=[]
probe_108_infos_head=[]

probe_126_images_head=[]
probe_126_infos_head=[]

probe_144_images_head=[]
probe_144_infos_head=[]

probe_162_images_head=[]
probe_162_infos_head=[]

probe_180_images_head=[]
probe_180_infos_head=[]

for pim_head, pin_head in zip(probe_images_head,probe_infos_head):
    if(pin_head[1]=='000'):
        probe_000_images_head.append(pim_head)
        probe_000_infos_head.append(pin_head)
    if(pin_head[1]=='018'):
        probe_018_images_head.append(pim_head)
        probe_018_infos_head.append(pin_head)
    if(pin_head[1]=='036'):
        probe_036_images_head.append(pim_head)
        probe_036_infos_head.append(pin_head)
    if(pin_head[1]=='054'):
        probe_054_images_head.append(pim_head)
        probe_054_infos_head.append(pin_head)
    if(pin_head[1]=='072'):
        probe_072_images_head.append(pim_head)
        probe_072_infos_head.append(pin_head)
    if(pin_head[1]=='090'):
        probe_090_images_head.append(pim_head)
        probe_090_infos_head.append(pin_head)
    if(pin_head[1]=='108'):
        probe_108_images_head.append(pim_head)
        probe_108_infos_head.append(pin_head)
    if(pin_head[1]=='126'):
        probe_126_images_head.append(pim_head)
        probe_126_infos_head.append(pin_head)
    if(pin_head[1]=='144'):
        probe_144_images_head.append(pim_head)
        probe_144_infos_head.append(pin_head)
    if(pin_head[1]=='162'):
        probe_162_images_head.append(pim_head)
        probe_162_infos_head.append(pin_head)
    if(pin_head[1]=='180'):
        probe_180_images_head.append(pim_head)
        probe_180_infos_head.append(pin_head)
        

######################################################################## ALL GALERIES HEAD

All_gallery_angles_images_head=[]
All_gallery_angles_infos_head=[]

All_gallery_angles_images_head.append(gallery_000_images_head)
All_gallery_angles_infos_head.append(gallery_000_infos_head)

All_gallery_angles_images_head.append(gallery_018_images_head)
All_gallery_angles_infos_head.append(gallery_018_infos_head)

All_gallery_angles_images_head.append(gallery_036_images_head)
All_gallery_angles_infos_head.append(gallery_036_infos_head)

All_gallery_angles_images_head.append(gallery_054_images_head)
All_gallery_angles_infos_head.append(gallery_054_infos_head)

All_gallery_angles_images_head.append(gallery_072_images_head)
All_gallery_angles_infos_head.append(gallery_072_infos_head)

All_gallery_angles_images_head.append(gallery_090_images_head)
All_gallery_angles_infos_head.append(gallery_090_infos_head)

All_gallery_angles_images_head.append(gallery_108_images_head)
All_gallery_angles_infos_head.append(gallery_108_infos_head)

All_gallery_angles_images_head.append(gallery_126_images_head)
All_gallery_angles_infos_head.append(gallery_126_infos_head)

All_gallery_angles_images_head.append(gallery_144_images_head)
All_gallery_angles_infos_head.append(gallery_144_infos_head)

All_gallery_angles_images_head.append(gallery_162_images_head)
All_gallery_angles_infos_head.append(gallery_162_infos_head)

All_gallery_angles_images_head.append(gallery_180_images_head)
All_gallery_angles_infos_head.append(gallery_180_infos_head)
########################################################################### ALL PROBES HEAD

All_probe_angles_images_head=[]
All_probe_angles_infos_head=[]

All_probe_angles_images_head.append(probe_000_images_head)
All_probe_angles_infos_head.append(probe_000_infos_head)

All_probe_angles_images_head.append(probe_018_images_head)
All_probe_angles_infos_head.append(probe_018_infos_head)

All_probe_angles_images_head.append(probe_036_images_head)
All_probe_angles_infos_head.append(probe_036_infos_head)

All_probe_angles_images_head.append(probe_054_images_head)
All_probe_angles_infos_head.append(probe_054_infos_head)

All_probe_angles_images_head.append(probe_072_images_head)
All_probe_angles_infos_head.append(probe_072_infos_head)

All_probe_angles_images_head.append(probe_090_images_head)
All_probe_angles_infos_head.append(probe_090_infos_head)

All_probe_angles_images_head.append(probe_108_images_head)
All_probe_angles_infos_head.append(probe_108_infos_head)

All_probe_angles_images_head.append(probe_126_images_head)
All_probe_angles_infos_head.append(probe_126_infos_head)

All_probe_angles_images_head.append(probe_144_images_head)
All_probe_angles_infos_head.append(probe_144_infos_head)

All_probe_angles_images_head.append(probe_162_images_head)
All_probe_angles_infos_head.append(probe_162_infos_head)

All_probe_angles_images_head.append(probe_180_images_head)
All_probe_angles_infos_head.append(probe_180_infos_head)


######################################################### GALLERIES LEG
gallery_000_images_leg=[]
gallery_000_infos_leg=[]

gallery_018_images_leg=[]
gallery_018_infos_leg=[]

gallery_036_images_leg=[]
gallery_036_infos_leg=[]

gallery_054_images_leg=[]
gallery_054_infos_leg=[]

gallery_072_images_leg=[]
gallery_072_infos_leg=[]

gallery_090_images_leg=[]
gallery_090_infos_leg=[]

gallery_108_images_leg=[]
gallery_108_infos_leg=[]

gallery_126_images_leg=[]
gallery_126_infos_leg=[]

gallery_144_images_leg=[]
gallery_144_infos_leg=[]

gallery_162_images_leg=[]
gallery_162_infos_leg=[]

gallery_180_images_leg=[]
gallery_180_infos_leg=[]

for gim_leg, gin_leg in zip(gallery_images_leg,gallery_infos_leg):
    if(gin_leg[1]=='000'):
        gallery_000_images_leg.append(gim_leg)
        gallery_000_infos_leg.append(gin_leg)
    if(gin_leg[1]=='018'):
        gallery_018_images_leg.append(gim_leg)
        gallery_018_infos_leg.append(gin_leg)
    if(gin_leg[1]=='036'):
        gallery_036_images_leg.append(gim_leg)
        gallery_036_infos_leg.append(gin_leg)
    if(gin_leg[1]=='054'):
        gallery_054_images_leg.append(gim_leg)
        gallery_054_infos_leg.append(gin_leg)
    if(gin_leg[1]=='072'):
        gallery_072_images_leg.append(gim_leg)
        gallery_072_infos_leg.append(gin_leg)
    if(gin_leg[1]=='090'):
        gallery_090_images_leg.append(gim_leg)
        gallery_090_infos_leg.append(gin_leg)
    if(gin_leg[1]=='108'):
        gallery_108_images_leg.append(gim_leg)
        gallery_108_infos_leg.append(gin_leg)
    if(gin_leg[1]=='126'):
        gallery_126_images_leg.append(gim_leg)
        gallery_126_infos_leg.append(gin_leg)
    if(gin_leg[1]=='144'):
        gallery_144_images_leg.append(gim_leg)
        gallery_144_infos_leg.append(gin_leg)
    if(gin_leg[1]=='162'):
        gallery_162_images_leg.append(gim_leg)
        gallery_162_infos_leg.append(gin_leg)
    if(gin_leg[1]=='180'):
        gallery_180_images_leg.append(gim_leg)
        gallery_180_infos_leg.append(gin_leg)

################################################################### PROBES LEG
probe_000_images_leg=[]
probe_000_infos_leg=[]

probe_018_images_leg=[]
probe_018_infos_leg=[]

probe_036_images_leg=[]
probe_036_infos_leg=[]

probe_054_images_leg=[]
probe_054_infos_leg=[]

probe_072_images_leg=[]
probe_072_infos_leg=[]

probe_090_images_leg=[]
probe_090_infos_leg=[]

probe_108_images_leg=[]
probe_108_infos_leg=[]

probe_126_images_leg=[]
probe_126_infos_leg=[]

probe_144_images_leg=[]
probe_144_infos_leg=[]

probe_162_images_leg=[]
probe_162_infos_leg=[]

probe_180_images_leg=[]
probe_180_infos_leg=[]

for pim_leg, pin_leg in zip(probe_images_leg,probe_infos_leg):
    if(pin_leg[1]=='000'):
        probe_000_images_leg.append(pim_leg)
        probe_000_infos_leg.append(pin_leg)
    if(pin_leg[1]=='018'):
        probe_018_images_leg.append(pim_leg)
        probe_018_infos_leg.append(pin_leg)
    if(pin_leg[1]=='036'):
        probe_036_images_leg.append(pim_leg)
        probe_036_infos_leg.append(pin_leg)
    if(pin_leg[1]=='054'):
        probe_054_images_leg.append(pim_leg)
        probe_054_infos_leg.append(pin_leg)
    if(pin_leg[1]=='072'):
        probe_072_images_leg.append(pim_leg)
        probe_072_infos_leg.append(pin_leg)
    if(pin_leg[1]=='090'):
        probe_090_images_leg.append(pim_leg)
        probe_090_infos_leg.append(pin_leg)
    if(pin_leg[1]=='108'):
        probe_108_images_leg.append(pim_leg)
        probe_108_infos_leg.append(pin_leg)
    if(pin_leg[1]=='126'):
        probe_126_images_leg.append(pim_leg)
        probe_126_infos_leg.append(pin_leg)
    if(pin_leg[1]=='144'):
        probe_144_images_leg.append(pim_leg)
        probe_144_infos_leg.append(pin_leg)
    if(pin_leg[1]=='162'):
        probe_162_images_leg.append(pim_leg)
        probe_162_infos_leg.append(pin_leg)
    if(pin_leg[1]=='180'):
        probe_180_images_leg.append(pim_leg)
        probe_180_infos_leg.append(pin_leg)
        

######################################################################## ALL GALERIES LEG

All_gallery_angles_images_leg=[]
All_gallery_angles_infos_leg=[]

All_gallery_angles_images_leg.append(gallery_000_images_leg)
All_gallery_angles_infos_leg.append(gallery_000_infos_leg)

All_gallery_angles_images_leg.append(gallery_018_images_leg)
All_gallery_angles_infos_leg.append(gallery_018_infos_leg)

All_gallery_angles_images_leg.append(gallery_036_images_leg)
All_gallery_angles_infos_leg.append(gallery_036_infos_leg)

All_gallery_angles_images_leg.append(gallery_054_images_leg)
All_gallery_angles_infos_leg.append(gallery_054_infos_leg)

All_gallery_angles_images_leg.append(gallery_072_images_leg)
All_gallery_angles_infos_leg.append(gallery_072_infos_leg)

All_gallery_angles_images_leg.append(gallery_090_images_leg)
All_gallery_angles_infos_leg.append(gallery_090_infos_leg)

All_gallery_angles_images_leg.append(gallery_108_images_leg)
All_gallery_angles_infos_leg.append(gallery_108_infos_leg)

All_gallery_angles_images_leg.append(gallery_126_images_leg)
All_gallery_angles_infos_leg.append(gallery_126_infos_leg)

All_gallery_angles_images_leg.append(gallery_144_images_leg)
All_gallery_angles_infos_leg.append(gallery_144_infos_leg)

All_gallery_angles_images_leg.append(gallery_162_images_leg)
All_gallery_angles_infos_leg.append(gallery_162_infos_leg)

All_gallery_angles_images_leg.append(gallery_180_images_leg)
All_gallery_angles_infos_leg.append(gallery_180_infos_leg)
########################################################################### ALL PROBES LEG

All_probe_angles_images_leg=[]
All_probe_angles_infos_leg=[]

All_probe_angles_images_leg.append(probe_000_images_leg)
All_probe_angles_infos_leg.append(probe_000_infos_leg)

All_probe_angles_images_leg.append(probe_018_images_leg)
All_probe_angles_infos_leg.append(probe_018_infos_leg)

All_probe_angles_images_leg.append(probe_036_images_leg)
All_probe_angles_infos_leg.append(probe_036_infos_leg)

All_probe_angles_images_leg.append(probe_054_images_leg)
All_probe_angles_infos_leg.append(probe_054_infos_leg)

All_probe_angles_images_leg.append(probe_072_images_leg)
All_probe_angles_infos_leg.append(probe_072_infos_leg)

All_probe_angles_images_leg.append(probe_090_images_leg)
All_probe_angles_infos_leg.append(probe_090_infos_leg)

All_probe_angles_images_leg.append(probe_108_images_leg)
All_probe_angles_infos_leg.append(probe_108_infos_leg)

All_probe_angles_images_leg.append(probe_126_images_leg)
All_probe_angles_infos_leg.append(probe_126_infos_leg)

All_probe_angles_images_leg.append(probe_144_images_leg)
All_probe_angles_infos_leg.append(probe_144_infos_leg)

All_probe_angles_images_leg.append(probe_162_images_leg)
All_probe_angles_infos_leg.append(probe_162_infos_leg)

All_probe_angles_images_leg.append(probe_180_images_leg)
All_probe_angles_infos_leg.append(probe_180_infos_leg)




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


for i in range(len(All_probe_angles_images_head)):
    for j in range(len(All_gallery_angles_images_head)):
        
        g = tf.Graph()
        with g.as_default():
               
            leg_modal = load_model('weights/MobileNet_Leg.ckpt')
            head_modal = load_model('weights/MobileNet_Head.ckpt')

            for layer in leg_modal.layers:
                layer.trainable = False
                layer._name = 'leg_' + layer._name
            for layer2 in head_modal.layers:
                layer2.trainable = False
                layer2._name = 'head_' + layer2._name

            leg_output = leg_modal.get_layer('leg_global_average_pooling2d').output
            head_output = head_modal.get_layer('head_global_average_pooling2d').output
            fused_output = Concatenate(name='fusion_layer')([leg_output, head_output])
            net = Model(inputs=[leg_modal.input, head_modal.input], outputs=fused_output, name='two_stream')
            
            test_features = extract_feature(All_gallery_angles_images_leg[j], All_gallery_angles_images_head[j], net)
            query_features = extract_feature(All_probe_angles_images_leg[i], All_probe_angles_images_head[i], net)

        result = sess.run(tensor, {query_t: query_features, test_t: test_features})
        
        Probe_num=len(All_probe_angles_images_head[i])
        tmp_probe_infos=[]
        for k in range(len(All_probe_angles_images_head[i])):
            tmp_probe_infos.append(0)
            
        indexes=[]
        for m in range(result.shape[0]):
            index=np.argmax(result[m])
            tmp_probe_infos[m]= All_gallery_angles_infos_head[j][index] 
            indexes.append(index)
            
        true_count=0
        for n in range(len(tmp_probe_infos)):
            if tmp_probe_infos[n][0]==All_probe_angles_infos_head[i][n][0]: 
                true_count=true_count+1

                
        
        success=100*true_count/Probe_num
        result_table[i][j]=success
        success=0
        
