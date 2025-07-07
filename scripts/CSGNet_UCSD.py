import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import random as rn
import tensorflow as tf

# set current working directory
cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)

# =============================================================================
#  For reprodocable results
# =============================================================================
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import keras, glob
from keras.preprocessing import image as kImage
from keras.utils.data_utils import get_file
from sklearn.utils import compute_class_weight
from scripts.CSGNet_module import CSGNet_module
import gc

if keras.__version__!= '2.2.2' or tf.__version__!='1.10.0' or sys.version_info[0]<3:
    print('We implemented using [keras v2.2.2, tensorflow-gpu v1.10.0, python v3.6.13], other versions than these may cause errors somehow!\n')

def getData(train_dir, dataset_dir):
    # given ground-truths, load inputs  
    Y_list = glob.glob(os.path.join(train_dir,'*.png'))
    X_list = glob.glob(os.path.join(dataset_dir, 'in', '*.jpg'))
    if len(Y_list)<=0 or len(X_list)<=0:
        raise ValueError('System cannot find the dataset path or ground-truth path. Please give the correct path.')
    
    X_list_temp = []
    for i in range(len(Y_list)):
        Y_name = os.path.basename(Y_list[i])
        Y_name = Y_name.split('.')[0]
        Y_name = Y_name.split('gt')[1]
        for j in range(len(X_list)):
            X_name = os.path.basename(X_list[j])
            X_name = X_name.split('.')[0]
            X_name = X_name.split('_')[1]
            if (int(Y_name) == int(X_name)):
                X_list_temp.append(X_list[j])
                break
    X_list = X_list_temp
    if len(X_list)!=len(Y_list):
        raise ValueError('The number of X_list and Y_list must be equal.')

    # load training data
    X = []
    Y = []
    for i in range(len(X_list)):
        x = kImage.load_img(X_list[i])
        x = kImage.img_to_array(x)
        X.append(x)
        
        x = kImage.load_img(Y_list[i], grayscale = True)
        x = kImage.img_to_array(x)
        shape = x.shape
        x /= 255.0
        x = x.reshape(-1)
        x = x.reshape(shape)
        x = np.floor(x)
        Y.append(x)
    X = np.asarray(X)
    Y = np.asarray(Y)

    # We do not consider temporal data
    idx = list(range(X.shape[0]))
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]
    
    cls_weight_list = []
    for i in range(Y.shape[0]):
        y = Y[i].reshape(-1)
        lb = np.unique(y) #  0., 1
        cls_weight = compute_class_weight('balanced', lb , y)
        class_0 = cls_weight[0]
        class_1 = cls_weight[1] if len(lb)>1 else 1.0
        
        cls_weight_dict = {0:class_0, 1: class_1}
        cls_weight_list.append(cls_weight_dict)
    cls_weight_list = np.asarray(cls_weight_list)
    
    return [X, Y, cls_weight_list]
def train(data, scene, mdl_path, vgg_weights_path):
    ### hyper-params
    lr = 1e-5
    val_split = 0
    max_epoch = 200
    batch_size = 1
    ###
    
    img_shape = data[0][0].shape #(height, width, channel)
    model = CSGNet_module(lr, img_shape, scene, vgg_weights_path)
    model = model.initModel('UCSD')
    
    # make sure that training input shape equals to model output
    input_shape = (img_shape[0], img_shape[1])
    output_shape = (model.output._keras_shape[1], model.output._keras_shape[2])
    assert input_shape==output_shape, 'Given input shape:' + str(input_shape) + ', but your model outputs shape:' + str(output_shape)
    
    redu = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto')
    model.fit({'net_input':data[0]}, {'output_x':data[1]}, 
                validation_split=val_split, 
                epochs=max_epoch, batch_size=batch_size, 
                callbacks=[redu], verbose=1, 
                class_weight={'output_x':data[2]}, shuffle = True)
    
    model.save(mdl_path)
    del model, data, redu


# =============================================================================
# Main func
# =============================================================================
dataset = [
            'birds', 'boats', 'bottle', 'chopper', 'cyclists', 
            'flock', 'freeway', 'hockey', 'jump', 'landing', 
            'ocean', 'peds', 'rain', 'skiing', 'surf', 'surfers', 
            'traffic', 'zodiac'
        ]

main_dir = os.path.join('.', 'CSGNet')
vgg_weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
if not os.path.exists(vgg_weights_path):
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    vgg_weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP, cache_subdir='models',
                                file_hash='6d6bbae143d832006294945121d1f1fc')

# =============================================================================
num_split = 10 # either 10% or 20% or 50% training split
# =============================================================================
assert num_split in [10,20,50], 'num_split is incorrect.'
main_mdl_dir = os.path.join(main_dir, 'UCSD', f'models{num_split}')
if not os.path.exists(main_mdl_dir):
    os.makedirs(main_mdl_dir)

for scene in dataset:
    print(f'Training ->>> {scene}')
    
    train_dir = os.path.join('.', 'training_sets', f'UCSD_train{num_split}', scene)
    dataset_dir = os.path.join('.', 'datasets', 'UCSD_dataset', scene)
    data = getData(train_dir, dataset_dir)
    
    mdl_path = os.path.join(main_mdl_dir, f'mdl_{scene}.h5')
    train(data, scene, mdl_path, vgg_weights_path)
    
    del data
    gc.collect()