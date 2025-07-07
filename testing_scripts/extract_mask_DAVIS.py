import warnings
warnings.filterwarnings('ignore')

import os
import sys
import glob
import numpy as np
import gc

from keras.models import load_model
from keras.preprocessing import image as kImage
from keras.preprocessing.image import save_img
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.CSGNet_module import loss, acc, loss2, acc2
from scripts.my_upsampling_2d import MyUpSampling2D

def checkFrame(X_list):
    img = kImage.load_img(X_list[0])
    img = kImage.img_to_array(img).shape # (480,720,3)
    num_frames = len(X_list) # 7000
    max_frames = 800 # max frames to slice
    if(len(X_list)>max_frames):
        print(f'\t- Total Frames: {num_frames}')
        num_chunks = num_frames/max_frames
        num_chunks = int(np.ceil(num_chunks)) # 2.5 => 3 chunks
        start = 0
        end = max_frames
        m = [0]* num_chunks
        for i in range(num_chunks): # 5
            m[i] = range(start, end) # m[0,1500], m[1500, 3000], m[3000, 4500]
            start = end # 1500, 3000, 4500 
            if (num_frames - start > max_frames): # 1500, 500, 0
                end = start + max_frames # 3000
            else:
                end = start + (num_frames- start) # 2000 + 500, 2500+0
        print(f'\t- Slice to: {m}')
        del img, X_list
        return [True, m]
    del img, X_list
    return [False, None]
def generateData(scene_input_path, X_list):
    X = []
    print('\n\t- Loading frames:')
    for i in range(0, len(X_list)):
        img = kImage.load_img(X_list[i])
        x = kImage.img_to_array(img)
        X.append(x)

        sys.stdout.write('\b' * len(str(i)))
        sys.stdout.write('\r')
        sys.stdout.write(str(i+1))
    X = np.asarray(X)
    del img, x, X_list
    return [X]
def getFiles(scene_input_path):
    inlist = glob.glob(os.path.join(scene_input_path,'*.jpg'))
    return np.asarray(inlist)


scene_list = [
              'bear', 'blackswan', 'bmx-bumps', 'bmx-trees', 'boat', 'breakdance', 'breakdance-flare', 'bus', 'camel', 'car-roundabout', 
              'car-shadow', 'car-turn', 'cows', 'dance-jump', 'dance-twirl', 'dog', 'dog-agility', 'drift-chicane', 'drift-straight', 'drift-turn', 
              'elephant', 'flamingo', 'goat', 'hike', 'hockey', 'horsejump-high', 'horsejump-low', 'kite-surf', 'kite-walk', 'libby', 
              'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 'motocross-jump', 'motorbike', 'paragliding', 'paragliding-launch', 'parkour', 'rhino', 
              'rollerblade', 'scooter-black', 'scooter-gray', 'soapbox', 'soccerball', 'stroller', 'surf', 'swing', 'tennis', 'train'
              ]

num_frames = 10
raw_dataset_dir = './datasets/DAVIS/JPEGImages/480p'
main_mdl_dir = os.path.join('./CSGNet/DAVIS', f'models{num_frames}')
results_dir = os.path.join('./CSGNet/DAVIS', f'results{num_frames}')

for scene in scene_list:
    print(f'\n->>> {scene}')
    mask_dir = os.path.join(results_dir, scene)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    
    scene_input_path = os.path.join(raw_dataset_dir, scene)
    X_list = getFiles(scene_input_path)
    if (X_list is None):
        raise ValueError('X_list is None')
    results = checkFrame(X_list) 

    mdl_path = os.path.join(main_mdl_dir, f'mdl_{scene}.h5')
    model = load_model(mdl_path, compile=False, custom_objects=
                        {'InstanceNormalization': InstanceNormalization, 
                        'MyUpSampling2D': MyUpSampling2D, 
                        'loss':loss, 'acc':acc})

    # if large numbers of frames, slice it
    if(results[0]): 
        for rangeee in results[1]: # for each slice
            slice_X_list =  X_list[rangeee]
            data= generateData(scene_input_path, slice_X_list, scene)
            Y_proba = model.predict(data, batch_size=1, verbose=1)
            del data
            
            # filter out
            shape = Y_proba.shape
            Y_proba = Y_proba.reshape([shape[0],-1])
            if (len(idx)>0): # if have non-ROI
                for i in range(len(Y_proba)): # for each frames
                    Y_proba[i][idx] = 0. # set non-ROI pixel to black
                    
            Y_proba = Y_proba.reshape([shape[0], shape[1], shape[2],shape[3]])

            prev = 0
            print('\n- Saving frames:')
            for i in range(shape[0]):
                fname = os.path.basename(slice_X_list[i]).replace('jpg','png')
                x = Y_proba[i]
                save_img(os.path.join(mask_dir, fname), x)
                sys.stdout.write('\b' * prev)
                sys.stdout.write('\r')
                s = str(i+1)
                sys.stdout.write(s)
                prev = len(s)
            del Y_proba, slice_X_list
    else: # otherwise, no need to slice
        data = generateData(scene_input_path, X_list)
        Y_proba = model.predict(data, batch_size=1, verbose=1)
        del data

        shape = Y_proba.shape
        Y_proba = Y_proba.reshape([shape[0],-1])
        Y_proba = Y_proba.reshape([shape[0], shape[1], shape[2], shape[3]])
        
        prev = 0
        print('\n- Saving frames:')
        for i in range(shape[0]):
            fname = os.path.basename(X_list[i]).replace('jpg','png')
            x = Y_proba[i]
            save_img(os.path.join(mask_dir, fname), x)
            sys.stdout.write('\b' * prev)
            sys.stdout.write('\r')
            s = str(i+1)
            sys.stdout.write(s)
            prev = len(s)
        del Y_proba
    del X_list, results
del model
gc.collect()
