import os, pickle, random, sys, shutil
from PIL import Image
import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
from tensorflow.contrib.framework import arg_scope

batch_size=256
save_path='./NTvsTInet2'


filesTI=[]
filesNT=[]
ntAns=[]
tiAns=[]

filesNT=os.listdir('DATA/test/NT/')
totNTNum=len(filesNT)
for i in range(totNTNum):
    filesNT[i]='DATA/test/NT/'+filesNT[i]

for k in range(totNTNum):
    ntAns.append([1, 0])

filesTI=os.listdir('DATA/test/TI/')
totTINum=len(filesTI)
for i in range(totTINum):
    filesTI[i]='DATA/test/TI/'+filesTI[i]

for k in range(totTINum):
    tiAns.append([0, 1])

totFiles=filesNT+filesTI
totAns=ntAns+tiAns
totFiles=np.array(totFiles)
totAns=np.array(totAns)

index=[]
for i in range(len(totFiles)):
    index.append(i)
random.shuffle(index)
totFiles=totFiles[index]
totAns=totAns[index]

totImages=[]
for i in range(len(totFiles)):
    im = Image.open(totFiles[i])
    totImages.append(np.array(im))

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

turn=len(totFiles)//batch_size

if tf.gfile.Exists('DATA/test/NTbutTI/'):
    tf.gfile.DeleteRecursively('DATA/test/NTbutTI/')
tf.gfile.MakeDirs('DATA/test/NTbutTI/')

if tf.gfile.Exists('DATA/test/TIbutNT/'):
    tf.gfile.DeleteRecursively('DATA/test/TIbutNT/')
tf.gfile.MakeDirs('DATA/test/TIbutNT/')

for j in range(turn):
    if j==turn:
        if len(totFiles)%batch_size:
            train_batch = totImages[j * batch_size:]
            ans_batch = totAns[j * batch_size:]
        else:
            continue
    else:
        train_batch = totImages[j * batch_size:(j + 1) * batch_size]
        ans_batch = totAns[j * batch_size:(j + 1) * batch_size]
    acc,result=sess.run((caccuracy,ccorrect_prediction), feed_dict={cx: train_batch, ct: ans_batch, train_mode: False})
    indexForFalse=np.where(result==False)
    print(indexForFalse[0])
    for i in range(len(indexForFalse[0])):
        falseFile=totFiles[j * batch_size+indexForFalse[0][i]]
        falseFileAns=totAns[j * batch_size+indexForFalse[0][i]]
        print(falseFile)
        if falseFileAns[0]==1:
            shutil.copy(falseFile, 'DATA/test/NTbutTI/')
        else:
            shutil.copy(falseFile, 'DATA/test/TIbutNT/')
    print(acc)
