# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A binary to evaluate Inception on the ImageNet data set.

Note that using the supplied pre-trained inception checkpoint, the eval should
achieve:
  precision @ 1 = 0.7874 recall @ 5 = 0.9436 [50000 examples]

See the README.md for more details.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
#sys.path.insert(0, '/home/jajman/PycharmProjects/hepatoCA')

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
from tensorflow.contrib.framework import arg_scope


from datetime import datetime
import os
import random
import sys
import threading

import numpy as np

import openslide, time, os, pickle, shutil
from openslide import OpenSlide, OpenSlideError, deepzoom

import matplotlib.pyplot as plt

from PIL import Image, ImageStat

save_path='./NTvsTInet2'
def getFileName(fileName):
    lenFIleName = len(fileName)
    nameEnd = 0
    nameStart = 0
    isDot=False
    for i in reversed(range(lenFIleName)):
        if fileName[i] == '.':
            if isDot:
                continue
            else:
                nameEnd = i
                isDot=True
        if fileName[i] == '/':
            nameStart = i + 1
            break
    return fileName[nameStart:nameEnd]

def inboundVal(val,min,max):
    if val<=min:
        return min
    if val>=max:
        return max







with tf.name_scope("CNN"):
    num_filters1 = 12

    cx = tf.placeholder(tf.float32, [None, 360,360,3])
    x_image = tf.reshape(cx, [-1, 360,360, 3])

    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, num_filters1],
                                              stddev=0.1))
    h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                           strides=[1, 1, 1, 1], padding='SAME')

    b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
    h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)

    h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    num_filters2 = 24

    W_conv2 = tf.Variable(
        tf.truncated_normal([5, 5, num_filters1, num_filters2],
                            stddev=0.1))
    h_conv2 = tf.nn.conv2d(h_pool1, W_conv2,
                           strides=[1, 1, 1, 1], padding='SAME')

    b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
    h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)

    h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    num_filters3 = 24

    W_conv3 = tf.Variable(
        tf.truncated_normal([5, 5, num_filters2, num_filters3],
                            stddev=0.1))
    h_conv3 = tf.nn.conv2d(h_pool2, W_conv3,
                           strides=[1, 1, 1, 1], padding='SAME')

    b_conv3 = tf.Variable(tf.constant(0.1, shape=[num_filters3]))
    h_conv3_cutoff = tf.nn.relu(h_conv3 + b_conv3)

    h_pool3 = tf.nn.max_pool(h_conv3_cutoff, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')


    #print(h_pool2)
    h_pool3_flat = tf.reshape(h_pool3, [-1, 45 * 45 * num_filters3])

    num_units1 = 45 * 45 * num_filters3
    num_units2 = 512
    num_units3 = 128
    final_output_size = 2
    keep_prob = 0.7
    train_mode = tf.placeholder(tf.bool, name='train_mode')
    xavier_init = tf.contrib.layers.xavier_initializer()
    bn_params = {
        'is_training': train_mode,
        'decay': 0.9,
        'updates_collections': None
    }

    # We can build short code using 'arg_scope' to avoid duplicate code
    # same function with different arguments
    with arg_scope([fully_connected],
                   activation_fn=tf.nn.relu,
                   weights_initializer=xavier_init,
                   biases_initializer=None,
                   normalizer_fn=batch_norm,
                   normalizer_params=bn_params
                   ):
        hidden_layer1 = fully_connected(h_pool3_flat, num_units2, scope="h1")
        h1_drop = dropout(hidden_layer1, keep_prob, is_training=train_mode)
        hidden_layer2 = fully_connected(h1_drop, num_units3, scope="h2")
        h2_drop = dropout(hidden_layer2, keep_prob, is_training=train_mode)
        hypothesis = fully_connected(h2_drop, final_output_size, activation_fn=None, scope="hypothesis")

    cp = tf.nn.softmax(hypothesis)
    cp = cp + 0.000000001
    ct = tf.placeholder(tf.float32, [None, 2])
    closs = -tf.reduce_sum(ct * tf.log(cp))
    ctrain_step = tf.train.AdamOptimizer(0.0001).minimize(closs)
    ccorrect_prediction = tf.equal(tf.argmax(cp, 1), tf.argmax(ct, 1))
    caccuracy = tf.reduce_mean(tf.cast(ccorrect_prediction, tf.float32))

all_vars = tf.all_variables()
s_var_num=len(all_vars)



svar=all_vars[:s_var_num]
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
sckpt=tf.train.get_checkpoint_state(save_path)
spath=sckpt.model_checkpoint_path
tf.train.Saver(svar).restore(sess, sckpt.model_checkpoint_path)

def makeSplitData(folderName,fileNameHere,dirName):

    print(fileNameHere)
    splitNum=0
    curFolder=dirName +folderName+'/'
    svsFile=fileNameHere+'.svs'
    slide = OpenSlide(curFolder +svsFile)
    magVal=slide.properties['aperio.AppMag']
    mppVal=slide.properties['aperio.MPP']
    zoomed = deepzoom.DeepZoomGenerator(slide, tile_size=360, overlap=0, limit_bounds=False)
    levelCount=int(zoomed.level_count)
    maxD = slide.level_dimensions[0]
    maxY = maxD[1]
    maxX = maxD[0]
    levelTiles = zoomed.level_tiles
    if magVal=='40':
        targetLevel=levelCount-2
        totTiles=levelTiles[targetLevel]
        maxY=int(maxD[1]/2)
        maxX=int(maxD[0]/2)
    elif magVal=='20' and mppVal[:3]=='0.2':
        targetLevel = levelCount - 2
        totTiles = levelTiles[targetLevel]
        maxY = int(maxD[1] / 2)
        maxX = int(maxD[0] / 2)
    elif magVal=='20':
        targetLevel = levelCount - 1
        totTiles = levelTiles[targetLevel]
        maxY = int(maxD[1])
        maxX = int(maxD[0])
    else:
        print('mag is not 20 or 40!!!')
        return
    train_batch = []
    train_tile=[]
    train_ij=[]
    batch_num=0
    for i in range(totTiles[0]-1):
        for j in range(totTiles[1]-1):
            if i%10==0 and j%10==0:
                print('('+str(i)+','+str(j)+')')
            tile = zoomed.get_tile(targetLevel, (i, j))
            train_tile.append(tile)
            tileStartX=i*360
            tileEndX=(i+1)*360
            tileStartY = j * 360
            tileEndY = (j + 1) * 360
            batch_num+=1
            train_batch.append(np.array(tile))
            train_ij.append((i,j))
            if batch_num==256 or (i==(totTiles[0]-2) and j==(totTiles[1]-2)):
                result = sess.run((cp),
                                  feed_dict={cx: train_batch, train_mode: False})
                for k in range(len(result)):
                    if result[k][1] > 0.5:
                        train_tile[k].save(
                            curFolder+ '/TX/' + fileNameHere + '-' + str(train_ij[k][0]) + '-' + str(
                                train_ij[k][1]) + '.jpg')
                        splitNum += 1
                    else:
                        train_tile[k].save(
                            curFolder + '/NT/' + fileNameHere + '-' + str(train_ij[k][0]) + '-' + str(
                                train_ij[k][1]) + '.jpg')
                        splitNum += 1
                batch_num=0
                train_batch=[]
                train_ij=[]
                train_tile=[]






    return splitNum


def main(unused_argv=None):
    dirName = '/media/jajman/NewVolume/TCGA_colon_MSI/'

    files = os.listdir(dirName)
    files = sorted(files)
    fileNum = len(files)
    fileCount = 0
    for i in files:
        if tf.gfile.Exists(dirName+i+'/TX/'):
            tf.gfile.DeleteRecursively(dirName+i+'/TX/')
        tf.gfile.MakeDirs(dirName+i+'/TX/')
        if tf.gfile.Exists(dirName+i+'/NT/'):
            tf.gfile.DeleteRecursively(dirName+i+'/NT/')
        tf.gfile.MakeDirs(dirName+i+'/NT/')
        fileCount += 1
        print(str(fileCount) + '/' + str(fileNum))
        print(i)
        files2 = os.listdir(dirName + i)
        fileName = ''
        totSplit=0
        for j in files2:
            if j.endswith('.svs'):
                fileName = getFileName(j)
                totSplit += makeSplitData(i, fileName, dirName)
                print('total number')
                print(totSplit)


if __name__ == '__main__':
  tf.app.run()
