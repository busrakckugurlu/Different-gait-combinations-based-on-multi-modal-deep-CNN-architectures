
import numpy as np
import random
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Concatenate
from tools.create_train_test_data import load_testing_data_gei
from tools.load_gallery_probe import load_testing_data_stack



gallery_images_gei, gallery_infos_gei, probe_images_gei, probe_infos_gei = load_testing_data_gei("data/Casia_gait/gei")
gallery_images_stck, gallery_infos_stck, probe_images_stck, probe_infos_stck = load_testing_data_stack('data/10_pieces_OF/test')



######################################################################

def extract_feature(images_list1, images_list2,net):
    
  features = []
  for img1,img2 in zip(images_list1,images_list2):
    img_one = img1[random.randint(0, 9)]
    of_x= np.expand_dims(img_one, axis=0)
    gei_x= np.expand_dims(img2, axis=0)
    
    two_stream_x = [of_x, gei_x]
    
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
gallery_000_images_stck=[]
gallery_000_infos_stck=[]

gallery_018_images_stck=[]
gallery_018_infos_stck=[]

gallery_036_images_stck=[]
gallery_036_infos_stck=[]

gallery_054_images_stck=[]
gallery_054_infos_stck=[]

gallery_072_images_stck=[]
gallery_072_infos_stck=[]

gallery_090_images_stck=[]
gallery_090_infos_stck=[]

gallery_108_images_stck=[]
gallery_108_infos_stck=[]

gallery_126_images_stck=[]
gallery_126_infos_stck=[]

gallery_144_images_stck=[]
gallery_144_infos_stck=[]

gallery_162_images_stck=[]
gallery_162_infos_stck=[]

gallery_180_images_stck=[]
gallery_180_infos_stck=[]

for gim_stck, gin_stck in zip(gallery_images_stck,gallery_infos_stck):
    if(gin_stck[1]=='000'):
        gallery_000_images_stck.append(gim_stck)
        gallery_000_infos_stck.append(gin_stck)
    if(gin_stck[1]=='018'):
        gallery_018_images_stck.append(gim_stck)
        gallery_018_infos_stck.append(gin_stck)
    if(gin_stck[1]=='036'):
        gallery_036_images_stck.append(gim_stck)
        gallery_036_infos_stck.append(gin_stck)
    if(gin_stck[1]=='054'):
        gallery_054_images_stck.append(gim_stck)
        gallery_054_infos_stck.append(gin_stck)
    if(gin_stck[1]=='072'):
        gallery_072_images_stck.append(gim_stck)
        gallery_072_infos_stck.append(gin_stck)
    if(gin_stck[1]=='090'):
        gallery_090_images_stck.append(gim_stck)
        gallery_090_infos_stck.append(gin_stck)
    if(gin_stck[1]=='108'):
        gallery_108_images_stck.append(gim_stck)
        gallery_108_infos_stck.append(gin_stck)
    if(gin_stck[1]=='126'):
        gallery_126_images_stck.append(gim_stck)
        gallery_126_infos_stck.append(gin_stck)
    if(gin_stck[1]=='144'):
        gallery_144_images_stck.append(gim_stck)
        gallery_144_infos_stck.append(gin_stck)
    if(gin_stck[1]=='162'):
        gallery_162_images_stck.append(gim_stck)
        gallery_162_infos_stck.append(gin_stck)
    if(gin_stck[1]=='180'):
        gallery_180_images_stck.append(gim_stck)
        gallery_180_infos_stck.append(gin_stck)

################################################################### PROBES STACK
probe_000_images_stck=[]
probe_000_infos_stck=[]

probe_018_images_stck=[]
probe_018_infos_stck=[]

probe_036_images_stck=[]
probe_036_infos_stck=[]

probe_054_images_stck=[]
probe_054_infos_stck=[]

probe_072_images_stck=[]
probe_072_infos_stck=[]

probe_090_images_stck=[]
probe_090_infos_stck=[]

probe_108_images_stck=[]
probe_108_infos_stck=[]

probe_126_images_stck=[]
probe_126_infos_stck=[]

probe_144_images_stck=[]
probe_144_infos_stck=[]

probe_162_images_stck=[]
probe_162_infos_stck=[]

probe_180_images_stck=[]
probe_180_infos_stck=[]

for pim_stck, pin_stck in zip(probe_images_stck,probe_infos_stck):
    if(pin_stck[1]=='000'):
        probe_000_images_stck.append(pim_stck)
        probe_000_infos_stck.append(pin_stck)
    if(pin_stck[1]=='018'):
        probe_018_images_stck.append(pim_stck)
        probe_018_infos_stck.append(pin_stck)
    if(pin_stck[1]=='036'):
        probe_036_images_stck.append(pim_stck)
        probe_036_infos_stck.append(pin_stck)
    if(pin_stck[1]=='054'):
        probe_054_images_stck.append(pim_stck)
        probe_054_infos_stck.append(pin_stck)
    if(pin_stck[1]=='072'):
        probe_072_images_stck.append(pim_stck)
        probe_072_infos_stck.append(pin_stck)
    if(pin_stck[1]=='090'):
        probe_090_images_stck.append(pim_stck)
        probe_090_infos_stck.append(pin_stck)
    if(pin_stck[1]=='108'):
        probe_108_images_stck.append(pim_stck)
        probe_108_infos_stck.append(pin_stck)
    if(pin_stck[1]=='126'):
        probe_126_images_stck.append(pim_stck)
        probe_126_infos_stck.append(pin_stck)
    if(pin_stck[1]=='144'):
        probe_144_images_stck.append(pim_stck)
        probe_144_infos_stck.append(pin_stck)
    if(pin_stck[1]=='162'):
        probe_162_images_stck.append(pim_stck)
        probe_162_infos_stck.append(pin_stck)
    if(pin_stck[1]=='180'):
        probe_180_images_stck.append(pim_stck)
        probe_180_infos_stck.append(pin_stck)
        

######################################################################## ALL GALERIES STACK

All_gallery_angles_images_stck=[]
All_gallery_angles_infos_stck=[]

All_gallery_angles_images_stck.append(gallery_000_images_stck)
All_gallery_angles_infos_stck.append(gallery_000_infos_stck)

All_gallery_angles_images_stck.append(gallery_018_images_stck)
All_gallery_angles_infos_stck.append(gallery_018_infos_stck)

All_gallery_angles_images_stck.append(gallery_036_images_stck)
All_gallery_angles_infos_stck.append(gallery_036_infos_stck)

All_gallery_angles_images_stck.append(gallery_054_images_stck)
All_gallery_angles_infos_stck.append(gallery_054_infos_stck)

All_gallery_angles_images_stck.append(gallery_072_images_stck)
All_gallery_angles_infos_stck.append(gallery_072_infos_stck)

All_gallery_angles_images_stck.append(gallery_090_images_stck)
All_gallery_angles_infos_stck.append(gallery_090_infos_stck)

All_gallery_angles_images_stck.append(gallery_108_images_stck)
All_gallery_angles_infos_stck.append(gallery_108_infos_stck)

All_gallery_angles_images_stck.append(gallery_126_images_stck)
All_gallery_angles_infos_stck.append(gallery_126_infos_stck)

All_gallery_angles_images_stck.append(gallery_144_images_stck)
All_gallery_angles_infos_stck.append(gallery_144_infos_stck)

All_gallery_angles_images_stck.append(gallery_162_images_stck)
All_gallery_angles_infos_stck.append(gallery_162_infos_stck)

All_gallery_angles_images_stck.append(gallery_180_images_stck)
All_gallery_angles_infos_stck.append(gallery_180_infos_stck)
########################################################################### ALL PROBES STACK

All_probe_angles_images_stck=[]
All_probe_angles_infos_stck=[]

All_probe_angles_images_stck.append(probe_000_images_stck)
All_probe_angles_infos_stck.append(probe_000_infos_stck)

All_probe_angles_images_stck.append(probe_018_images_stck)
All_probe_angles_infos_stck.append(probe_018_infos_stck)

All_probe_angles_images_stck.append(probe_036_images_stck)
All_probe_angles_infos_stck.append(probe_036_infos_stck)

All_probe_angles_images_stck.append(probe_054_images_stck)
All_probe_angles_infos_stck.append(probe_054_infos_stck)

All_probe_angles_images_stck.append(probe_072_images_stck)
All_probe_angles_infos_stck.append(probe_072_infos_stck)

All_probe_angles_images_stck.append(probe_090_images_stck)
All_probe_angles_infos_stck.append(probe_090_infos_stck)

All_probe_angles_images_stck.append(probe_108_images_stck)
All_probe_angles_infos_stck.append(probe_108_infos_stck)

All_probe_angles_images_stck.append(probe_126_images_stck)
All_probe_angles_infos_stck.append(probe_126_infos_stck)

All_probe_angles_images_stck.append(probe_144_images_stck)
All_probe_angles_infos_stck.append(probe_144_infos_stck)

All_probe_angles_images_stck.append(probe_162_images_stck)
All_probe_angles_infos_stck.append(probe_162_infos_stck)

All_probe_angles_images_stck.append(probe_180_images_stck)
All_probe_angles_infos_stck.append(probe_180_infos_stck)




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
               
            of_modal = load_model('weights/MobileNet_OF.ckpt')
            gei_modal = load_model('weights/MobileNet_GEI.ckpt')

            for layer in of_modal.layers:
                layer.trainable = False
                layer._name = 'of_' + layer._name
            for layer2 in gei_modal.layers:
                layer2.trainable = False
                layer2._name = 'gei_' + layer2._name

            of_output = of_modal.get_layer('of_global_average_pooling2d').output
            gei_output = gei_modal.get_layer('gei_global_average_pooling2d').output
            fused_output = Concatenate(name='fusion_layer')([of_output, gei_output])
            net = Model(inputs=[of_modal.input, gei_modal.input], outputs=fused_output, name='two_stream')
            
            test_features = extract_feature(All_gallery_angles_images_stck[j], All_gallery_angles_images_gei[j], net)
            query_features = extract_feature(All_probe_angles_images_stck[i], All_probe_angles_images_gei[i], net)

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

