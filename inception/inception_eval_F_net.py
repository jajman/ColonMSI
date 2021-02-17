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
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.misc as misc
from datetime import datetime
import math, pickle
import os.path
import time, shutil, random


import numpy as np
import tensorflow as tf

from inception import image_processingN_RS
from inception import inception_model_net as inception


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/jajman/gastric_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/jajman/imagenet_train_plus37_8',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 0,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")


# Flags governing the data used for the eval.
'''tf.app.flags.DEFINE_integer('num_examples', fileNum,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")'''
tf.app.flags.DEFINE_string('subset', 'testing',
                           """Either 'validation' or 'train'.""")
'''tf.app.flags.DEFINE_integer('batch_size',  fileNum,
                           """Either 'validation' or 'train'.""")'''

def removeSlash(filePath):
  for i in range(len(filePath)):
    if filePath[i]=='/':
      filePath=filePath[:i]+'.'+filePath[i+1:]
  return filePath

def getFileName(fileName):
    lenFIleName = len(fileName)
    nameEnd = len(fileName)
    nameStart = 0
    isDot=False
    isNameStart=False
    for i in reversed(range(lenFIleName)):
        if fileName[i] == '.':
            if isDot:
                continue
            else:
                nameEnd = i
                isDot=True
        if fileName[i] == '/':
            if isNameStart==False:
                isNameStart=True
                nameEnd=i
                continue
            else:
                nameStart = i + 1
                break
    return fileName[nameStart:nameEnd]

def _eval_once(saver, summary_writer,  summary_op,logits_softmax,dataDir,xy_op,net,numTotShards):
  """Runs Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_1_op: Top 1 op.
    top_5_op: Top 5 op.
    summary_op: Summary op.
  """

  '''if not os.path.exists(dataDir +  '.bin'):
    dicForAll={}
  else:
    with open(dataDir +  '.bin','rb') as f:
      dicForAll=pickle.load(f)'''
  dicForAll = {}
  listForAll=[]
  L2All=[]
  finalList=[]
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if os.path.isabs(ckpt.model_checkpoint_path):
        # Restores from checkpoint with absolute path.
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        # Restores from checkpoint with relative path.
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                         ckpt.model_checkpoint_path))

      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/imagenet_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Successfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    totLogit=[]
    totXY = []
    totNet=[]
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      #num_iter = int(math.ceil(num_examples / batch_size))
      # Counts the number of correct predictions.
      count_top_1 = 0.0
      count_top_5 = 0.0
      #total_sample_count = num_iter * batch_size
      step = 0

      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
      while step < numTotShards and not coord.should_stop():
          print(datetime.now())
          srcDir = '/home/jajman/testingDATA/' + dataDir + '/image' + str(step) + '/'
          print(srcDir)
          moveFile = srcDir + 'testing-00000-of-00001'

          if tf.gfile.Exists(FLAGS.data_dir):
              tf.gfile.DeleteRecursively(FLAGS.data_dir)
          tf.gfile.MakeDirs(FLAGS.data_dir)
          shutil.copy(moveFile, FLAGS.data_dir)
          time.sleep(0.1)

          step += 1
          xy_val = sess.run([xy_op])
          #totXY.append(xy_val[0])
          logitVal = sess.run([logits_softmax])
          #totLogit.append(logitVal[0])
          time.sleep(0.1)
          xy_val = sess.run([xy_op])
          totXY.append(xy_val[0])
          logitVal = sess.run([logits_softmax])
          totLogit.append(logitVal[0])
          netVal=sess.run([net])
          totNet.append(netVal[0])
          print('done')


      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))

      summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    toWrite=''
    print(len(totNet))
    totNum=0
    for i in range(len(totNet)):
        for j in range(len(totNet[i])):
          L1 =totLogit[i][j][1]
          L2 =totLogit[i][j][2]
          tot = L1 + L2
          ratio = 1 / tot
          L2 = L2 * ratio
          listForAll.append([dataDir, totXY[i][j], totNet[i][j], totLogit[i][j][2]])
          L2All.append(totLogit[i][j][2])
          #if L2>0.95:
          #if L2<0.25:
            #print(L2)
            #totNum+=1
            #dicForAll[totXY[i][j]]=totNet[i][j]


          #dicForAll[totXY[i][j]] = totLogit[i][j][1]
    '''print('---------------------------------------------------')
    print(totNum)
    print('---------------------------------------------------')'''

    '''L2All.sort(reverse=True)
    #L2All.sort()
    L2SortedUniqueList=[]
    for i in L2All:
        if i not in L2SortedUniqueList:
            L2SortedUniqueList.append(i)

    allNum=0
    for i in listForAll:
        if i[3] in L2SortedUniqueList[:500]:
            finalList.append(i)
            allNum += 1
            if allNum == 500:
                break'''

    '''for i in listForAll:
        print(i[1])
    print('---------------------------------------------------')'''
    random.shuffle(listForAll)
    '''for i in listForAll:
        print(i[1])
    print('---------------------------------------------------')'''
    for i in range(1000):
        finalList.append(listForAll[i])


    #fileToSave=getFileName(FLAGS.data_dir)
    #fileToSave=removeSlash(fileToSave)

    with open(dataDir +  '.bin', 'wb') as f:
        pickle.dump(finalList, f)

'''dirName='test/adeno/'
files=os.listdir(dirName)
images=[]
for file in files:
    totImage = misc.imread(dirName + file)
    images.append(totImage)
images=np.array(images)
images=images/256
images=images-0.5
images=images*2.0
images1=tf.convert_to_tensor(images,tf.float32)'''

def evaluate(dataset,dataDir, numTotShards):


  batch_size = FLAGS.batch_size
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels from the dataset.
    images, labels,xyCoord = image_processingN_RS.inputs(dataset,batch_size=batch_size)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = dataset.num_classes() + 1

    # Build a Graph that computes the logits predictions from the
    # inference model.
    net, logits, _ = inception.inference(images, num_classes)
    logits_softmax = tf.nn.softmax(logits)
    #logits1, _ = inception.inference(images1, num_classes)
    xy_op = xyCoord
    # Calculate predictions.
    '''top_1_op = tf.nn.in_top_k(logits, labels, 1)
    top_5_op = tf.nn.in_top_k(logits, labels, 5)'''

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

    while True:
      _eval_once(saver, summary_writer, summary_op,logits_softmax,dataDir,xy_op,net,numTotShards)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

