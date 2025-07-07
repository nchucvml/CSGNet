import keras
from keras.models import Model
from keras.layers import Input, Dropout, Activation, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Cropping2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import concatenate, add, multiply
from .my_upsampling_2d import MyUpSampling2D
from .instance_normalization import InstanceNormalization
import keras.backend as K
import tensorflow as tf

def loss(y_true, y_pred):
    void_label = -1.
    y_pred = K.reshape(y_pred, [-1])
    y_true = K.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
def acc(y_true, y_pred):
    void_label = -1.
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
def loss2(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
def acc2(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

class CSGNet_module(object):
    def __init__(self, lr, img_shape, scene, vgg_weights_path):
        self.lr = lr
        self.img_shape = img_shape
        self.scene = scene
        self.vgg_weights_path = vgg_weights_path
        self.method_name = 'CSGNet'
        
    def triple_VGG16(self, x): 
        x2_input = Conv2D(3, (3, 3), padding='same', dilation_rate=2)(x)
        x2_input = MaxPooling2D((2, 2), strides=(2, 2), name='x2_input')(x2_input)

        x4_input = Conv2D(3, (3, 3), padding='same', dilation_rate=2)(x2_input)
        x4_input = MaxPooling2D((2, 2), strides=(2, 2), name='x4_input')(x4_input)

        share_block1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')
        share_block1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')

        share_block2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')
        share_block2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')

        share_block3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')
        share_block3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')
        share_block3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')

        share_block4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')
        share_block4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')
        share_block4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')

        # First VGG
        x = share_block1_1(x)
        x = share_block1_2(x)
        a = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
        x = share_block2_1(x)
        x = share_block2_2(x)
        b = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        x_down = x

        x = share_block3_1(x)
        x = share_block3_2(x)
        x = share_block3_3(x)
    
        x = share_block4_1(x)
        x = Dropout(0.5, name='dr1')(x)
        x = share_block4_2(x)
        x = Dropout(0.5, name='dr2')(x)
        x = share_block4_3(x)
        x = Dropout(0.5, name='dr3')(x)

        # Second VGG
        x2 = share_block1_1(x2_input)
        x2 = share_block1_2(x2)
        x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_2')(x2)
        x2_down = x2

        x2 = concatenate([x2_down, x_down], axis=-1, name='cat_x2')
        x2 = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x2)
    
        x2 = share_block2_1(x2)
        x2 = share_block2_2(x2)

        x2 = share_block3_1(x2)
        x2 = share_block3_2(x2)

        # Third VGG
        x4 = concatenate([x2_down, x_down, x4_input], axis=-1, name='cat_x4')
        x4 = Conv2D(3, (1, 1), strides=(1, 1), padding='same')(x4)

        x4 = share_block1_1(x4)
        x4 = share_block1_2(x4)

        x4 = share_block2_1(x4)
        x4 = share_block2_2(x4)
        
        return x, a, b, x2, x4
    def M_SEM(self, x,x_2,x_4):
        d1 = Conv2D(64, (3, 3), padding='same')(x)        
        y = concatenate([x, d1], axis=-1, name='cat4')
        y = Activation('relu')(y)
        d4 = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(y)
        y = concatenate([x, d4], axis=-1, name='cat8')
        y = Activation('relu')(y)
        d8 = Conv2D(64, (3, 3), padding='same', dilation_rate=4)(y)
        y = concatenate([x, d8], axis=-1, name='cat16')
        y = Activation('relu')(y)
        d16 = Conv2D(64, (3, 3), padding='same', dilation_rate=8)(y)    

        maxpool_x1 = MaxPooling2D((2, 2), strides=(1, 1), name='max_pool1',padding='same')(x)
        maxpool_x1 = Conv2D(64, (1, 1), padding='same')(maxpool_x1)

        maxpool_x2 = MaxPooling2D((2, 2), strides=(1, 1), name='max_pool2',padding='same')(x_2)
        maxpool_x2 = Conv2D(64, (1, 1), padding='same')(maxpool_x2)

        maxpool_x4 = MaxPooling2D((2, 2), strides=(1, 1), name='max_pool4',padding='same')(x_4)
        maxpool_x4 = Conv2D(64, (1, 1), padding='same')(maxpool_x4)

        
        d1_x_2 = Conv2D(64, (3, 3), padding='same')(x_2)
        y_x_2 = concatenate([x, d1_x_2], axis=-1, name='cat4_x_2')
        y_x_2 = Activation('relu')(y_x_2)
        d4_x_2 = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(y_x_2)     
        y_x_2 = concatenate([x, d4_x_2], axis=-1, name='cat8_x_2')
        y_x_2 = Activation('relu')(y_x_2)
        d8_x_2 = Conv2D(64, (3, 3), padding='same', dilation_rate=4)(y_x_2)
        y_x_2 = concatenate([x, d8_x_2], axis=-1, name='cat16_x_2')
        y_x_2 = Activation('relu')(y_x_2)
        d16_x_2 = Conv2D(64, (3, 3), padding='same', dilation_rate=8)(y_x_2) 
        
        d1_x_4 = Conv2D(64, (3, 3), padding='same')(x_4)     
        y_x_4 = concatenate([x, d1_x_4], axis=-1, name='cat4_x_4')
        y_x_4 = Activation('relu')(y_x_4)
        d4_x_4 = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(y_x_4)     
        y_x_4 = concatenate([x, d4_x_4], axis=-1, name='cat8_x_4')
        y_x_4 = Activation('relu')(y_x_4)
        d8_x_4 = Conv2D(64, (3, 3), padding='same', dilation_rate=4)(y_x_4)
        y_x_4 = concatenate([x, d8_x_4], axis=-1, name='cat16_x_4')
        y_x_4 = Activation('relu')(y_x_4)
        d16_x_4 = Conv2D(64, (3, 3), padding='same', dilation_rate=8)(y_x_4)

        
        x_r = concatenate([d1, d4, d8, d16,maxpool_x1], axis=-1,name='xr')
        x_r = InstanceNormalization()(x_r)
        x_r = Activation('relu')(x_r)
        x_r = SpatialDropout2D(0.25)(x_r)
        
        x_r_2 = concatenate([d1_x_2, d4_x_2, d8_x_2 ,d16_x_2,maxpool_x2], axis=-1,name='xr_2')
        x_r_2 = InstanceNormalization()(x_r_2)
        x_r_2 = Activation('relu')(x_r_2)
        x_r_2 = SpatialDropout2D(0.25)(x_r_2)   
        
        x_r_4 = concatenate([d1_x_4, d4_x_4, d8_x_4 ,d16_x_4,maxpool_x4], axis=-1,name='xr_4')
        x_r_4 = InstanceNormalization()(x_r_4)
        x_r_4 = Activation('relu')(x_r_4)
        x_r_4 = SpatialDropout2D(0.25)(x_r_4)           
        
        return x_r, x_r_2, x_r_4    
    def decoder(self,x,a,b,x_size2,x_size4):
        a = GlobalAveragePooling2D()(a)
        b = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(b)
        b = GlobalAveragePooling2D()(b)

        # First MSI
        shared_conv_4 = Conv2D(64, (3,3), padding='same')            
        x = shared_conv_4(x)
        x_size2 = shared_conv_4(x_size2)
        x_size4 = shared_conv_4(x_size4)

        x = InstanceNormalization()(x)
        x = Activation('relu')(x)        
        
        x_size2 = InstanceNormalization()(x_size2)
        x_size2 = Activation('relu')(x_size2)

        x_size4 = InstanceNormalization()(x_size4)
        x_size4 = Activation('relu')(x_size4)   
        
        x_4 = concatenate([x, x_size2,x_size4], axis=-1, name='cat_x_4_decoder')

        # use low resolution for middle attention
        x_4 = Conv2D(64, (1,1), padding='same')(x_4)

        x_att = multiply([x, x_4])
        x = add([x, x_att])

        x_att2 = multiply([x_size2, x_4])
        x_size2 = add([x_size2, x_att2])

                
        # output
        x1 = multiply([x, b])
        x = add([x, x1])
        
        x_size22 = multiply([x_size2, b])
        x_size2 = add([x_size2, x_size22])
               
        x = UpSampling2D(size=(2, 2))(x)
        x_size2 = UpSampling2D(size=(2, 2))(x_size2)
        
        
        # Second MSI
        shared_conv_2 = Conv2D(64, (3,3), padding='same')
        x = shared_conv_2(x)            
        x_size2 = shared_conv_2(x_size2) 

        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        
                       
        x_size2 = InstanceNormalization()(x_size2)
        x_size2 = Activation('relu')(x_size2)
        
        
        x_2 = concatenate([x, x_size2], axis=-1, name='3-size-concate_2')

        # use middle resolution for high attention
        x_2 = Conv2D(64, (1,1), padding='same')(x_2) 

        x_att11 = multiply([x, x_2])
        x = add([x, x_att11])

        x2 = multiply([x, a])
        x = add([x, x2]) 
         
        x = UpSampling2D(size=(2, 2))(x)
        
        shared_conv_1 = Conv2D(64, (3,3), padding='same')
         
        x = shared_conv_1(x) 
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)     
         
        return x
    
    def initModel(self, dataset_name):
        assert dataset_name in ['CDnet', 'UCSD', 'DAVIS'], 'dataset_name must be either one in ["CDnet", "UCSD", "DAVIS"]]'
        assert len(self.img_shape)==3
        h, w, d = self.img_shape
        
        net_input = Input(shape=(h, w, d), name='net_input')
        vgg_output = self.triple_VGG16(net_input)
        model = Model(inputs=net_input, outputs=vgg_output, name='model')       
        model.load_weights(self.vgg_weights_path, by_name=True)
        
        unfreeze_layers = ['block4_conv1','block4_conv2', 'block4_conv3']
        for layer in model.layers:
            if(layer.name not in unfreeze_layers):
                layer.trainable  = False
        x,a,b,x2_b,x4_a = model.output

        # pad in case of CDnet2014
        if dataset_name=='CDnet':
            x1_ups = {'streetCornerAtNight':(0,1), 'tramStation':(1,0), 'turbulence2':(1,0)}
            for key, val in x1_ups.items():
                if self.scene==key:
                    # upscale by adding number of pixels to each dim.
                    x = MyUpSampling2D(size=(1,1), num_pixels=val, method_name = self.method_name)(x)
                    x2_b=MyUpSampling2D(size=(1,1), num_pixels=val, method_name = self.method_name)(x2_b)
                    x4_a=MyUpSampling2D(size=(1,1), num_pixels=val, method_name = self.method_name)(x4_a)
                    break
        x,x_size2_SEM,x_size4_SEM = self.M_SEM(x,x2_b,x4_a)
        x = self.decoder(x,a,b,x_size2_SEM,x_size4_SEM)
            
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x) 
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        
        tosetname = 'output_x'
        if dataset_name == 'CDnet':
            rename_scene_list = ['tramCrossroad_1fps', 'bridgeEntry', 'fluidHighway', 'streetCornerAtNight', 
                                 'tramStation', 'twoPositionPTZCam', 'turbulence2', 'turbulence3']
            if(self.scene in rename_scene_list):
                tosetname = 'output_x_before'
        elif dataset_name == 'DAVIS':
            tosetname = 'output_x_before'
        x = Conv2D(1,1, padding='same', activation='sigmoid',name=tosetname)(x) 
        
        # pad in case of CDnet2014
        if dataset_name == 'CDnet':
            if(self.scene=='tramCrossroad_1fps'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0), method_name=self.method_name,name='output_x')(x)
            elif(self.scene=='bridgeEntry'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,2), method_name=self.method_name,name='output_x')(x)
            elif(self.scene=='fluidHighway'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0), method_name=self.method_name,name='output_x')(x)
            elif(self.scene=='streetCornerAtNight'): 
                x = MyUpSampling2D(size=(1,1), num_pixels=(1,0), method_name=self.method_name)(x)
                x = Cropping2D(cropping=((0, 0),(0, 1)),name='output_x')(x)
            elif(self.scene=='tramStation'):  
                x = Cropping2D(cropping=((1, 0),(0, 0)),name='output_x')(x)
            elif(self.scene=='twoPositionPTZCam'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(0,2), method_name=self.method_name,name='output_x')(x)
            elif(self.scene=='turbulence2'):
                x = Cropping2D(cropping=((1, 0),(0, 0)))(x)
                x = MyUpSampling2D(size=(1,1), num_pixels=(0,1), method_name=self.method_name,name='output_x')(x)
            elif(self.scene=='turbulence3'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0), method_name=self.method_name,name='output_x')(x)
        elif dataset_name == 'DAVIS':
            x = MyUpSampling2D(size=(1,1), num_pixels=(0,2), method_name=self.method_name,name='output_x')(x)
        vision_model = Model(inputs=net_input, outputs=x, name='vision_model')
        opt = keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon=1e-08, decay=0.)

        # Since UCSD has no void label, we do not need to filter out
        if dataset_name == 'UCSD':
            c_loss = loss2
            c_acc = acc2
        else:
            c_loss = loss
            c_acc = acc

        losses = {'output_x':loss}
        lossWeights = {'output_x':1}
        accs = {'output_x':c_acc}
 
        vision_model.compile(loss=losses,loss_weights=lossWeights, optimizer=opt, metrics=accs)
        return vision_model