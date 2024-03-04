
import numpy as np
import random
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.backend import set_session
from keras.models import Model
from keras.models import load_model
from keras.layers import Concatenate
from load_gallery_probe import load_testing_data_gei



gallery_images_gei, gallery_infos_gei, probe_images_gei, probe_infos_gei = load_testing_data_gei('data/OutdoorGei/OutdoorGait_gei_train_test/gei/test','bg_bg')
gallery_images_hconl, gallery_infos_hconl, probe_images_hconl, probe_infos_hconl = load_testing_data_gei('data/HConL/test','bg_bg')





def extract_feature(images_list1,images_list2, net):
    
  features = []
  
  for img1,img2 in zip(images_list1,images_list2):
      hconl_x= np.expand_dims(img1, axis=0)
      gei_x= np.expand_dims(img2, axis=0)
      
      two_stream_x = [hconl_x, gei_x]
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
    
    hconl_modal = load_model('weights/MobileNet_Outdoor_HConL.ckpt')
    gei_modal = load_model('weights/MobileNet_Outdoor_GEI.ckpt')
    
    
    for layer in hconl_modal.layers:
        layer.trainable = False
        layer._name = 'hconl_' + layer._name
    for layer2 in gei_modal.layers:
        layer2.trainable = False
        layer2._name = 'gei_' + layer2._name
    

    spatial_output = hconl_modal.get_layer('hconl_global_average_pooling2d').output
    temporal_output = gei_modal.get_layer('gei_global_average_pooling2d').output
    
    
    fused_output = Concatenate(name='fusion_layer')([spatial_output, temporal_output])
    net = Model(inputs=[hconl_modal.input, gei_modal.input], outputs=fused_output, name='two_stream')
    
    test_features = extract_feature(gallery_images_hconl,gallery_images_gei, net)
    query_features = extract_feature(probe_images_hconl,probe_images_gei, net)
    
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


