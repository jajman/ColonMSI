import os, pickle, random, sys
from PIL import Image
import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
from tensorflow.contrib.framework import arg_scope

batch_size=128
save_path='./NTvsTInet2'
train_r=0.95
val_r=0.05

filesNT=os.listdir('DATA/train/NT/')
totNTNum=len(filesNT)
for i in range(totNTNum):
    filesNT[i]='DATA/train/NT/'+filesNT[i]

ntAns=[]
for k in range(totNTNum):
    ntAns.append([1, 0])



filesTI=os.listdir('DATA/train/TI/')
totTINum=len(filesTI)
for i in range(totTINum):
    filesTI[i]='DATA/train/TI/'+filesTI[i]

tiAns=[]
for k in range(totTINum):
    tiAns.append([0, 1])



#train, validation 용 data의 개수를 정한다.
nt_train_num=int(len(filesNT)*train_r)
nt_val_num=len(filesNT)-nt_train_num
ti_train_num=int(len(filesTI)*train_r)
ti_val_num=len(filesTI)-ti_train_num

#train, validation 용 data를 잘라서 얻는다.
nt_train = filesNT[0:nt_train_num]
ti_train = filesTI[0:ti_train_num]

nt_val = filesNT[nt_train_num:]
ti_val = filesTI[ti_train_num:]

nt_ans_train = ntAns[0:nt_train_num]
ti_ans_train = tiAns[0:ti_train_num]

nt_ans_val = ntAns[nt_train_num:]
ti_ans_val = tiAns[ti_train_num:]

nt_ans_train.extend(ti_ans_train)
nt_ans_val.extend(ti_ans_val)
nt_train.extend(ti_train)
nt_val.extend(ti_val)

#배열 data 처리를 용이하게 하기 위해 data list를 np.array로 변환한다.
nt_ans_train = np.array(nt_ans_train)
nt_ans_val = np.array(nt_ans_val)
nt_train = np.array(nt_train)
nt_val = np.array(nt_val)

index=[]
for i in range(len(nt_val)):
    index.append(i)
random.shuffle(index)
nt_val=nt_val[index]
nt_ans_val=nt_ans_val[index]

#앞의 것들은 실제 data가 아니라 파일 이름들이었을 뿐이므로 이제 validation data의 실제 조직 image를 읽어들여 np.array로 저장해 둔다.
nt_val_images=[]
for i in range(len(nt_val)):
    im = Image.open(nt_val[i])
    nt_val_images.append(np.array(im))

#train data에 대해서는 뒤에 실제 train 중에 읽어 들이고 일단 index만 만들어 둔다.
index=[]
for i in range(len(nt_train)):
    index.append(i)
turn=len(nt_train)//batch_size


#본 과제 수행을 위한 deep neural network의 구조를 다음과 같이 정의한다.
#########################################################################################
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
#########################################################################################


sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
saver=tf.train.Saver(max_to_keep=10)


if tf.gfile.Exists(save_path):
    tf.gfile.DeleteRecursively(save_path)
tf.gfile.MakeDirs(save_path)


#아무런 training이 이루어 지지 않은 default network의 결과물은 다음과 같이 한번 test 해 본다.
loss_val, acc_val = sess.run([closs, caccuracy],
                                 feed_dict={cx: nt_val_images[:500], ct: nt_ans_val[:500], train_mode: False})
print('pretrain , Loss: %f, Accuracy: %f'
              % ( loss_val, acc_val))


#자 실제 training을 시작한다.
time.sleep(2)
startTime = time.time()
maxAcc=0
minLoss=10000
logStr=''
print('turn for a full training session:' +str(turn))
for i in range(30):
    random.shuffle(index)
    for j in range(turn):
        if j%20==0:
            print(str(i)+' '+str(j))
        train_batch=nt_train[index[j * batch_size:(j + 1) * batch_size]]
        nt_train_images = []
        for k in range(len(train_batch)):
            im = Image.open(train_batch[k])
            nt_train_images.append(np.array(im))
        ans_batch=nt_ans_train[index[j * batch_size:(j + 1) * batch_size]]
        sess.run(ctrain_step, feed_dict={cx: nt_train_images, ct: ans_batch, train_mode: True})

    if True:

        loss_val, acc_val = sess.run([closs, caccuracy],
                                     feed_dict={cx: nt_val_images[:500], ct: nt_ans_val[:500], train_mode: False})
        if acc_val>maxAcc and i>1:
            maxAcc=acc_val
            minLoss = loss_val
            saver.save(sess, save_path + '/model-' + str(i) + '.cptk')
        if acc_val==maxAcc and i>2:
            if minLoss>loss_val:
                minLoss=loss_val
                saver.save(sess, save_path + '/model-' + str(i) + '.cptk')

        train_batch = nt_train[index[:500]]
        nt_train_images = []
        for k in range(len(train_batch)):
            im = Image.open(train_batch[k])

            rand = random.random()
            if rand < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            rand = random.random()
            if rand < 0.5:
                im = im.rotate(90)
            rand = random.random()
            if rand < 0.5:
                im = im.rotate(90)

            nt_train_images.append(np.array(im))
        ans_batch = nt_ans_train[index[:500]]

        loss_train, acc_train = sess.run([closs, caccuracy],
                                         feed_dict={cx: nt_train_images, ct: ans_batch, train_mode: False})
        print('Step: %d, Train Loss: %f, Train Accuracy: %f, Val Loss: %f, Val Accuracy: %f'
              % (i, loss_train, acc_train, loss_val, acc_val))
        logStr += 'Step: %d, Train Loss: %f, Train Accuracy: %f, Val Loss: %f, Val Accuracy: %f' % (
            i, loss_train, acc_train, loss_val, acc_val)
        logStr += '\n'



endTime = time.time()
print(str(endTime - startTime) + ' SEC')
