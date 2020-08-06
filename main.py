import tensorflow as tf

import keras

from segmentation_models import Unet

import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
# from keras import backend as keras
import tensorflow as tf
from keras.losses import *
from keras import regularizers
import scipy.io as sio
import cv2

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.set_session(sess)

# backbone in ['vgg16',
#             'vgg19',
#             'resnet18',
#             'resnet34',
#             'resnet50',
#             'resnet101',
#             'resnet152',
#             'resnext50',
#             'resnext101',
#             'densenet121',
#             'densenet169',
#             'densenet201',
#             'inceptionv3',
#             'inceptionresnetv2',
#             ]
'''
decoder_block_type:
    transpose or upsampling
'''
model = Unet(backbone_name='',
             encoder_weights=None,
             decoder_block_type='transpose',
             classes=1,
             input_shape=[256, 256, 1],
             l2_regular=0.0,
             l1_regular=0.00)  # l1_ and l2_regular for regularzation
                               # both are zero would achieve best performance


# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 0.0001
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)

def data_generator(data, batchsize):

    images = data['images']
    masks = data['masks']
    N = images.shape[0]
    while True:
        index = np.arange(N)
        np.random.shuffle(index)
        for i in range(0, N, batchsize):
            input = []
            target = []
            for j in range(i, i+batchsize):
                if j>=N:
                    break

                image = images[index[j], :, :].astype(np.float64) / 255 # normalization
                mask = masks[index[j], :, :].astype(np.float64)

                image = cv2.resize(image, (256, 256))
                mask = cv2.resize(mask, (256, 256)) # resize to 256*256

                r = np.random.random()
                if r > 0.5:
                    image = np.flip(image, axis=0)  # data argument
                    mask = np.flip(mask, axis=0)
                r = np.random.random()
                if r > 0.5:
                    image = np.flip(image, axis=1)
                    mask = np.flip(mask, axis=1)

                image = image[:, :, np.newaxis]
                mask = mask[:, :, np.newaxis]

                input.append(image)
                target.append(mask)
            input = np.array(input)
            target = np.array(target)
            yield input, target

def pre_processing():
    images = []
    masks = []
    files = os.listdir('')
    for file in files:
        name = os.path.join('', file)
        data = sio.loadmat(name)
        image = data['images']
        fluid = data['manualFluid']
        slices = image.shape[2]
        for slice in range(slices):
            mask = fluid[:, :, slice]
            if np.isnan(mask).any() == False:
                images.append(image[:, :, slice])
                masks.append(mask)
    images = np.array(images)
    masks = np.array(masks)
    masks = (masks!=0).astype(np.int)
    N = images.shape[0]
    # shuffle the order
    index = np.arange(N)
    np.random.shuffle(index)

    trainNum = int(N*0.7) # 70% for training
    valNum = int(N*0.2) # 20% for validation, and the remain 10% for test

    sio.savemat('Training.mat',{'images':images[0:trainNum, :, :], 'masks':masks[0:trainNum, :, :]})
    sio.savemat('Validation.mat',{'images':images[trainNum:trainNum+valNum, :, :], 'masks':masks[trainNum:trainNum+valNum, :, :]})
    sio.savemat('Test.mat',{'images':images[trainNum+valNum:N, :, :], 'masks':masks[trainNum+valNum:N, :, :]})



status = 'Training'  # training or test
if status == 'Training':

    if ( os.path.exists('Training.mat') and os.path.exists('Validation.mat') and os.path.exists('Test.mat') ) == False:
        pre_processing()

    import os, datetime

    nowTime = str(datetime.datetime.now())
    nowTime = nowTime.replace('-', '_')
    nowTime = nowTime.replace(':', '_')
    nowTime = nowTime.replace(' ', '_')
    nowTime = nowTime.replace('.', '_')

    os.makedirs(os.path.join('record', nowTime, 'models'))
    os.makedirs(os.path.join('record', nowTime, 'log'))   # make the directory to save the log and models

    batch = 2
    epochs = 100

    import math
    def step_decay(epoch):
        initial_lrate = 0
        drop = 0.9
        epochs_drop = 10
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate
    lrate = LearningRateScheduler(step_decay)
    # optimizer = Adam(learning_rate=0.00)
    optimizer = SGD(lr=0.0, momentum=0.95, decay=0.0, nesterov=False)
    model.compile(optimizer=optimizer,
                  loss=bce_dice_loss,
                  metrics=["binary_crossentropy", 'accuracy', dice_coef])

    tensorboard = TensorBoard(log_dir=os.path.join('', nowTime, 'log'), histogram_freq=0)
    checkpoint = ModelCheckpoint(filepath=os.path.join('', nowTime, 'models',
                                                               ''),
                                         monitor='val_acc', verbose=0, save_best_only=False, mode='auto', period=2)
    trainingData = sio.loadmat('Training.mat')
    validationData = sio.loadmat('Validation.mat')
    train_generator = data_generator(trainingData, batchsize=batch)
    validation_generator = data_generator(validationData, batchsize=batch)
    model.fit_generator(
                generator=train_generator,
                # steps_per_epoch=2,
                steps_per_epoch=200,
                epochs=epochs,
                max_queue_size=100,
                validation_data=validation_generator,
                validation_steps=50,
                callbacks=[tensorboard, checkpoint, lrate],
                workers=1,
                use_multiprocessing=False
            )

elif status == 'Test':

    path = os.path.join('', '')
    model.load_weights(os.path.join(path, '', ''))
    import scipy.io as sio

    testData = sio.loadmat('Training.mat')
    images = testData['images']
    masks = testData['masks']
    N = images.shape[0]
    for i in range(N):
        oriimage = images[i, :, :].astype(np.float64)/255
        s = oriimage.shape
        image = cv2.resize(oriimage, (256, 256))
        image = image[np.newaxis, :, :, np.newaxis]
        pred = model.predict(image)
        pred = pred[0, :, :, 0]
        finalPred = cv2.resize(pred, (s[1], s[0]))
        predMask = (finalPred>0.05).astype(np.int)
        import matplotlib.pyplot as plt

        r = oriimage.copy()
        g = oriimage.copy()
        b = oriimage.copy()
        r[masks[i, :, :]==1] = 1
        g[masks[i, :, :]==1] = 0
        b[masks[i, :, :]==1] = 0
        r = r[:, :, np.newaxis]
        g = g[:, :, np.newaxis]
        b = b[:, :, np.newaxis]
        plt.figure()
        plt.subplot(1,2,1)
        showImg = np.concatenate([r,g,b], axis=2)
        plt.imshow(showImg)
        plt.title('true mask')

        r = oriimage.copy()
        g = oriimage.copy()
        b = oriimage.copy()
        r[predMask==1] = 1
        g[predMask==1] = 0
        b[predMask==1] = 0
        r = r[:, :, np.newaxis]
        g = g[:, :, np.newaxis]
        b = b[:, :, np.newaxis]
        showImg = np.concatenate([r, g, b], axis=2)
        plt.subplot(1,2,2)
        plt.imshow(showImg)
        plt.title('pred mask')
        # plt.show()
        plt.savefig(os.path.join('', str(i)+'.png'), format='png')
        plt.close()

        # image, mask = sess.run(TestData)
        #
        # _, h,w,s = image.shape
        # Mask = []
        # for slice in range(h):
        #     sliceImg, _ = getPatch(image[0, :, :, :], mask[0, :, :, :], slice, direction=direction)
        #     sliceImg = sliceImg[np.newaxis, :, :, :].astype(np.float32)
        #     sliceMask = model.predict(sliceImg)
        #     Mask.append(sliceMask[0, :, :, 1])
        # Mask = np.array(Mask)
        # sio.savemat(os.path.join(path, 'predictions',str(i)+'.mat'), {'pred_mask':Mask, 'true_mask':mask[0, :, :, :], 'original_image':image[0, :, :, :]})
