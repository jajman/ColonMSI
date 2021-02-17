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
sys.path.insert(0, '/home/jajman/PycharmProjects/gastricCATissue')

import tensorflow as tf

from inception import inception_eval_F_net
from inception.gastric_data import GastricData

from datetime import datetime
import os
import random
import sys, shutil, time
import threading

import numpy as np


from PIL import Image, ImageStat


tf.app.flags.DEFINE_string('testing_directory', '/home/jajman/testing/',
                           'Training data directory')



tf.app.flags.DEFINE_integer('testing_shards', 1,
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_string('output_directory', '/home/jajman/testingDATA/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('num_threads', 1,
                            'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_string('labels_file', 'testing.txt', 'Labels file')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.


FLAGS = tf.app.flags.FLAGS





def main(unused_argv=None):
  dataDir= sys.argv[1]
  dataDir=dataDir.replace('=',' ')
  with open('/home/jajman/PycharmProjects/gastricCATissue/' +dataDir+ '-dir.txt', 'rt') as fp:
      lines=fp.readlines()
  for i in range(len(lines)):
      lines[i]=lines[i].strip()

  target_data_dir = '/home/jajman/target/'
  if tf.gfile.Exists(target_data_dir):
      tf.gfile.DeleteRecursively(target_data_dir)
  tf.gfile.MakeDirs(target_data_dir)
  shutil.copy('/home/jajman/testingDATA/'+dataDir+'/image0/testing-00000-of-00001', target_data_dir)


  tf.app.flags.DEFINE_string('data_dir', target_data_dir,
                               """Path to the processed data, i.e. """
                               """TFRecord of Example protos.""")

  dataset = GastricData(subset=FLAGS.subset)
  assert dataset.data_files()
  if tf.gfile.Exists(FLAGS.eval_dir):
      tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  # inception_eval.evaluate(dataset,sys.argv[1])

  inception_eval_F_net.evaluate(dataset, dataDir, len(lines))
  time.sleep(0.5)




if __name__ == '__main__':
  tf.app.run()
