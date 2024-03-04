
import numpy as np
import random
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.backend import set_session
from keras.models import Model
from keras.models import load_model
from keras.layers import Concatenate
from tools.load_gallery_probe import load_testing_data_gei



gallery_images_head, gallery_infos_head, probe_images_head, probe_infos_head = load_testing_data_gei('data/Cropped_Head/test','bg_bg')
gallery_images_leg, gallery_infos_leg, probe_images_leg, probe_infos_leg = load_testing_data_gei('data/Cropped_Leg/test','bg_bg')





def extract_feature(images_list1,images_list2, net):
    
  features = []
  
  for img1,img2 in zip(images_list1,images_list2):
      leg_x= np.expand_dims(img1, axis=0)
      head_x= np.expand_dims(img2, axis=0)
      
      two_stream_x = [leg_x, head_x]
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
    
    leg_modal = load_model('weights/MobileNet_Outdoor_Leg.ckpt')
    head_modal = load_model('weights/MobileNet_Outdoor_Head.ckpt')
    
    for layer in leg_modal.layers:
        layer.trainable = False
        layer._name = 'spatial_' + layer._name
    for layer2 in head_modal.layers:
        layer2.trainable = False
        layer2._name = 'temporal_' + layer2._name
    
    leg_output = leg_modal.get_layer('spatial_global_average_pooling2d').output
    head_output = head_modal.get_layer('temporal_global_average_pooling2d').output
    
    
    fused_output = Concatenate(name='fusion_layer')([leg_output, head_output])
    net = Model(inputs=[leg_modal.input, head_modal.input], outputs=fused_output, name='two_stream')
    
    test_features = extract_feature(gallery_images_leg,gallery_images_head, net)
    query_features = extract_feature(probe_images_leg,probe_images_head, net)
    
result = sess.run(tensor, {query_t: query_features, test_t: test_features})

Probe_num=len(probe_images_head)
tmp_probe_infos=[]
for k in range(len(probe_images_head)):
    tmp_probe_infos.append(0)
    
indexes=[]
for m in range(result.shape[0]):
    index=np.argmax(result[m])
    tmp_probe_infos[m]= gallery_infos_head[index] 
    indexes.append(index)
    
true_count=0
for n in range(len(tmp_probe_infos)):
    if tmp_probe_infos[n]==probe_infos_head[n]: 
        true_count=true_count+1

        

success=100*true_count/Probe_num          


