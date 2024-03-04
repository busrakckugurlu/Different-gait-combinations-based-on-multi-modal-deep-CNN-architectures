
import numpy as np
import random
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.backend import set_session
from keras.models import Model
from keras.models import load_model
from keras.layers import Concatenate
from tools.load_gallery_probe_gei import load_testing_data_gei
from tools.load_gallery_probe import load_testing_data_frame



gallery_images_gei, gallery_infos_gei, probe_images_gei, probe_infos_gei = load_testing_data_gei('data/OutdoorGei/OutdoorGait_gei_train_test/gei/test','bg_bg')
gallery_images_frame, gallery_infos_frame, probe_images_frame, probe_infos_frame = load_testing_data_frame('data/Siluet_elimineted/test','bg_bg')





def extract_feature(images_list1,images_list2, net):
    
  features = []
  
  for img1,img2 in zip(images_list1,images_list2):
      silh_x= np.expand_dims(img1, axis=0)
      gei_x= np.expand_dims(img2, axis=0)
      
      two_stream_x = [silh_x, gei_x]
      feature = net.predict(two_stream_x)
      features.append(np.squeeze(feature))
      
  return features




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




g = tf.Graph()
with g.as_default():
    
    silh_modal = load_model('weights/MobileNet_Outdoor_SH.ckpt')
    gei_modal = load_model('For_GEI_HConL_Head_Leg/model/weights/MobileNet_Outdoor_GEI.ckpt')
    
    for layer in silh_modal.layers:
        layer.trainable = False
        layer._name = 'silh_' + layer._name
    for layer2 in gei_modal.layers:
        layer2.trainable = False
        layer2._name = 'gei_' + layer2._name
    
    silh_output = silh_modal.get_layer('silh_global_average_pooling2d').output
    gei_output = gei_modal.get_layer('gei_global_average_pooling2d').output
    
    fused_output = Concatenate(name='fusion_layer')([silh_output, gei_output])
    net = Model(inputs=[silh_modal.input, gei_modal.input], outputs=fused_output, name='two_stream')
    
    test_features = extract_feature(gallery_images_frame,gallery_images_gei, net)
    query_features = extract_feature(probe_images_frame,probe_images_gei, net)
    
result = sess.run(tensor, {query_t: query_features, test_t: test_features})

Probe_num=len(probe_images_gei)
tmp_probe_infos=[]
for k in range(len(probe_images_gei)):
    tmp_probe_infos.append(0)
    
indexes=[]
for m in range(result.shape[0]):
    index=np.argmax(result[m])
    tmp_probe_infos[m]= gallery_infos_gei[index] 
    indexes.append(index)
    
true_count=0
for n in range(len(tmp_probe_infos)):
    if tmp_probe_infos[n]==probe_infos_gei[n]:   
        true_count=true_count+1
        

success=100*true_count/Probe_num          


