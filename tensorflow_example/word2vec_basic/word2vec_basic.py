# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import math
import os
import random
import sys
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
from common import logger
"""
Created on Tue Jun  4 17:31:11 2019

@author: Scarlett

https://blog.csdn.net/luozirong/article/details/73275847

https://github.com/tensorflow/tensorflow/edit/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
"""

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Basic word2vec example."""

data_index = 0


def word2vec_basic(log_dir):
  """Example of building, training and visualizing a word2vec model."""
  # Create the directory for TensorBoard variables if there is not.
  logger.debug('log_dir : {}'.format(log_dir))
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  # Step 1: Download the data.
  logger.debug('Step 1: Download the data ')
  print('Step 1')
  url = 'http://mattmahoney.net/dc/'

  # pylint: disable=redefined-outer-name
  def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    logger.debug('------------- enter maybe_download---------------')
    
    local_filename = os.path.join(gettempdir(), filename)
    if not os.path.exists(local_filename):
      local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                     local_filename)
    logger.debug('local_filename : {}'.format( local_filename))


    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
      print('Found and verified', filename)
      logger.debug('Found and verified : {}'.format( filename))

      
    else:
      print(statinfo.st_size)
      logger.debug('statinfo.st_size : {}'.format( statinfo.st_size))
      raise Exception('Failed to verify ' + local_filename +
                      '. Can you get to it with a browser?')
      
    logger.debug('------------- end maybe_download---------------')
    return local_filename

  filename = maybe_download('text8.zip', 31344016)

  # Read the data into a list of strings.
  def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    logger.debug('------------- enter read_data---------------')

    with zipfile.ZipFile(filename) as f:
      data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

  vocabulary = read_data(filename)
  print('Data size', len(vocabulary))
  logger.debug('Data size : {}'.format(len(vocabulary)))
  try:
      for i in range(10):
          logger.debug('{}'.format(vocabulary[i]))
  except Exception as e:
      logger.debug('can not print vocabulary : {} '.format(e))
  # Step 2: Build the dictionary and replace rare words with UNK token.
  
  logger.debug('Step 2: Build the dictionary and replace rare words with UNK token.')
  
  vocabulary_size = 50000

  def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    logger.debug('------------- enter build_dataset---------------')
    
    count = [['UNK', -1]]
    #   截取top 50000 的单词，其他词的index 为 0 
    count.extend(collections.Counter(words).most_common(n_words - 1))#  collections.Counter 返回统计集合，most_common 返回top N
    try:
        logger.debug('count[:5] : {}'.format(count[:5]))
        logger.debug('len count : {}'.format(len(count)))
    except Exception as e:
        logger.debug('can not print count cause : {}'.format(e))
    dictionary = {}
    flag_n = 0
    for word, _ in count:
      dictionary[word] = len(dictionary)
      flag_n += 1
      if flag_n < 10 :
          logger.debug('word : {} ; dictionary[word]: {} ;len(dictionary) :{}'.format(word,dictionary[word],len(dictionary)))
    data = []
    unk_count = 0
    for word in words:
      #     不在dictionary 中的word 会返回 0
      index = dictionary.get(word, 0)
      if unk_count<10:
          logger.debug('word : {} ; index : {}'.format(word,index))
      
      if index == 0:  # dictionary['UNK']
        unk_count += 1
        if unk_count<10:
            logger.debug('index == 0 ;unk_count : {}'.format(unk_count))
      data.append(index)
    count[0][1] = unk_count
    
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))#    逆向词典
    #   原dictionary:{'a': 5, 'b': 3, 'f': 2, 't': 2, 'd': 1, 's': 1, 'k': 1}
    #   reversed_distionary:{5: 'a', 3: 'b', 2: 't', 1: 'k'}
    logger.debug('------------- end build_dataset---------------')
    return data, count, dictionary, reversed_dictionary

  # Filling 4 global variables:
  # data - list of codes (integers from 0 to vocabulary_size-1).
  #   This is the original text but words are replaced by their codes
  # count - map of words(strings) to count of occurrences
  # dictionary - map of words(strings) to their codes(integers)
  # reverse_dictionary - maps codes(integers) to words(strings)
  data, count, unused_dictionary, reverse_dictionary = build_dataset(
      vocabulary, vocabulary_size)
  del vocabulary  # Hint to reduce memory.
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

  logger.debug('Most common words (+UNK), count[:5] : {} ;'.format( count[:5]))
  logger.debug('len(data) : {} ;Sample data : data[:10] : {} ; reverse_dictionary : {}'.format(len(data),data[:10], [reverse_dictionary[i] for i in data[:10]]))


  # Step 3: Function to generate a training batch for the skip-gram model.
  logger.debug('Step 3: Function to generate a training batch for the skip-gram model.')


  def generate_batch(batch_size, num_skips, skip_window):
    logger.debug('------------- enter generate_batch---------------')
      
    global data_index
    assert batch_size % num_skips == 0  #assert 检查条件，不符合就终止程序
    assert num_skips <= 2 * skip_window
    logger.debug('num_skips : {} ; skip_window : {} ; batch_size : {}'.format(num_skips,skip_window,batch_size))
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    logger.debug('batch : {}'.format(batch))
    logger.debug('labels : {} '.format(labels))
#    logger.debug('recheck batch and labels : ')
#    for n in batch:
#        logger.debug("{}".format(n))    
#    for n in labels:
#        logger.debug("{}".format(n))    
    logger.debug('span : {}'.format(span))
    logger.debug('buffer : {}'.format(buffer))
    logger.debug('data_index : {}'.format(data_index))

    if data_index + span > len(data):
      logger.debug('data_index + span > len(data)')
      data_index = 0
    buffer.extend(data[data_index:data_index + span])
    logger.debug('buffer excend  : {}'.format(buffer))
    
    data_index += span
    logger.debug(' after += :data_index : {}'.format(data_index))
    #  batch_size // num_skips 返回除法结果整数部分 
    logger.debug('***********i 循环 {} 次'.format(batch_size // num_skips))
    for i in range(batch_size // num_skips):
      logger.debug('--------------------i 第 {} 轮'.format(i))
      context_words = [w for w in range(span) if w != skip_window]
      logger.debug('context_words : {}'.format(context_words))
      words_to_use = random.sample(context_words, num_skips)
      logger.debug('words_to_use : {}'.format(words_to_use))
      logger.debug('********************j 循环 {} 次'.format(len(words_to_use)))
      #     enumerate() 返回序列各个元素及其下标（不指定则为index） 
      for j, context_word in enumerate(words_to_use):
        logger.debug('j 第 {} 轮'.format(j))
        logger.debug('i * num_skips + j : {}'.format(i * num_skips + j))
        logger.debug('skip_window : {} ; context_word : {} ;\n \t\t buffer[ {} ]: {} ;  buffer[ {} ] : {}'.format(skip_window,context_word,skip_window,buffer[skip_window],context_word,buffer[context_word]))
        batch[i * num_skips + j] = np.int32(buffer[skip_window])
        labels[i * num_skips + j, 0] = np.int32(buffer[context_word])

        logger.debug('batch[ {} ] : {}'.format(i * num_skips + j,batch[i * num_skips + j]))
        logger.debug('labels[ {}, 0] : {}'.format(i*num_skips + j,labels[i * num_skips + j, 0]))
      
      if data_index == len(data):
        logger.debug('data_index == len(data); buffer extend data[0:span] : {} '.format(data[0:span]))
        buffer.extend(data[0:span])
        data_index = span
      else:
        logger.debug('data_index != len(data); buffer append data[data_index] : {} '.format(data[data_index]))
        buffer.append(data[data_index])
        data_index += 1
      logger.debug('data_index : {}; buffer : {}'.format(data_index,buffer))
        
    # Backtrack a little bit to avoid skipping words in the end of a batch
    logger.debug('update data_index , len(data) = {}'.format(len(data)))
    #   % 求余
    data_index = (data_index + len(data) - span) % len(data)
#    logger.debug('{} data_index : {}'.format('-'*20,data_index))
#    logger.debug('1 batch.dtype : {} ; batch.shape : {}'.format(batch.dtype,batch.shape))
#    logger.debug('1 labels.dtype : {} ; labels.shape : {}'.format(labels.dtype,labels.shape))
#    if batch.dtype != 'int32':
#        batch.dtype = 'int32'
#    if labels.dtype != 'int32':
#        labels.dtype = 'int32' 
#    logger.debug('2 batch.dtype : {} ; batch.shape : {}'.format(batch.dtype,batch.shape))
#    logger.debug('2 labels.dtype : {} ; labels.shape : {}'.format(labels.dtype,labels.shape))      
    logger.debug('------------- end generate_batch---------------')
    return batch, labels


  def generate_batch_free(batch_size, num_skips, skip_window):
#    logger.debug('------------- enter generate_batch---------------')
      
    global data_index
    assert batch_size % num_skips == 0  #assert 检查条件，不符合就终止程序
    assert num_skips <= 2 * skip_window
#    logger.debug('num_skips : {} ; skip_window : {} ; batch_size : {}'.format(num_skips,skip_window,batch_size))
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
#    logger.debug('batch : {}'.format(batch))
#    logger.debug('labels : {} '.format(labels))
#    logger.debug('recheck batch and labels : ')
#    for n in batch:
#        logger.debug("{}".format(n))    
#    for n in labels:
#        logger.debug("{}".format(n))    
#    logger.debug('span : {}'.format(span))
#    logger.debug('buffer : {}'.format(buffer))
#    logger.debug('data_index : {}'.format(data_index))

    if data_index + span > len(data):
#      logger.debug('data_index + span > len(data)')
      data_index = 0
    buffer.extend(data[data_index:data_index + span])
#    logger.debug('buffer excend  : {}'.format(buffer))
    
    data_index += span
#    logger.debug(' after += :data_index : {}'.format(data_index))
    #  batch_size // num_skips 返回除法结果整数部分 
#    logger.debug('***********i 循环 {} 次'.format(batch_size // num_skips))
    for i in range(batch_size // num_skips):
#      logger.debug('--------------------i 第 {} 轮'.format(i))
      context_words = [w for w in range(span) if w != skip_window]
#      logger.debug('context_words : {}'.format(context_words))
      words_to_use = random.sample(context_words, num_skips)
#      logger.debug('words_to_use : {}'.format(words_to_use))
#      logger.debug('********************j 循环 {} 次'.format(len(words_to_use)))
      #     enumerate() 返回序列各个元素及其下标（不指定则为index） 
      for j, context_word in enumerate(words_to_use):
#        logger.debug('j 第 {} 轮'.format(j))
#        logger.debug('i * num_skips + j : {}'.format(i * num_skips + j))
#        logger.debug('skip_window : {} ; context_word : {} ;\n \t\t buffer[ {} ]: {} ;  buffer[ {} ] : {}'.format(skip_window,context_word,skip_window,buffer[skip_window],context_word,buffer[context_word]))
        batch[i * num_skips + j] = np.int32(buffer[skip_window])
        labels[i * num_skips + j, 0] = np.int32(buffer[context_word])

#        logger.debug('batch[ {} ] : {}'.format(i * num_skips + j,batch[i * num_skips + j]))
#        logger.debug('labels[ {}, 0] : {}'.format(i*num_skips + j,labels[i * num_skips + j, 0]))
#      
      if data_index == len(data):
#        logger.debug('data_index == len(data); buffer extend data[0:span] : {} '.format(data[0:span]))
        buffer.extend(data[0:span])
        data_index = span
      else:
#        logger.debug('data_index != len(data); buffer append data[data_index] : {} '.format(data[data_index]))
        buffer.append(data[data_index])
        data_index += 1
#      logger.debug('data_index : {}; buffer : {}'.format(data_index,buffer))
        
    # Backtrack a little bit to avoid skipping words in the end of a batch
#    logger.debug('update data_index , len(data) = {}'.format(len(data)))
    #   % 求余
    data_index = (data_index + len(data) - span) % len(data)
#    if batch.dtype != 'int32':
#        batch.dtype = 'int32'
#    if labels.dtype != 'int32':
#        labels.dtype = 'int32'     
#    logger.debug('{} data_index : {}'.format('-'*20,data_index))
#    logger.debug('------------- end generate_batch---------------')
    return batch, labels



  batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
  for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
          reverse_dictionary[labels[i, 0]])
    logger.debug('batch[ {} ] : {} ;\n reverse_dictionary[ {} ] :{} {} labels[{}, 0] : {} ; \n reverse_dictionary[{}]) : {}'.format(i,batch[i],
                batch[i], reverse_dictionary[batch[i]], '->',
                i, labels[i, 0], 
                labels[i, 0] , reverse_dictionary[labels[i, 0]]))    

  # Step 4: Build and train a skip-gram model.
  logger.debug('Step 4: Build and train a skip-gram model.')
  batch_size = 128
  embedding_size = 128  # Dimension of the embedding vector.
  skip_window = 1  # How many words to consider left and right.
  num_skips = 2  # How many times to reuse an input to generate a label.
  num_sampled = 64  # Number of negative examples to sample.

#   We pick a random validation set to sample nearest neighbors. Here we limit
#   the validation samples to the words that have a low numeric ID, which by
#   construction are also the most frequent. These 3 variables are used only for
#   displaying model accuracy, they don't affect calculation.
  valid_size = 16  # Random set of words to evaluate similarity on.
  valid_window = 100  # Only pick dev samples in the head of the distribution.
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)
  logger.debug('valid_examples : {} '.format(valid_examples))
    # np.random.choice 参数意思分别 是从a 中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布
    # a1 = np.random.choice(a=5, size=3, replace=False, p=None) replacement=False 代表放回抽样


  graph = tf.Graph()

  with graph.as_default():
    logger.debug('begin graph  ')
    # Input data.
    with tf.name_scope('inputs'):
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
#      train_inputs = tf.placeholder(tf.int32)
#      train_labels = tf.placeholder(tf.int32)      
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    #   测试默认是否运行 gpu
#    with tf.device('/cpu:0'):
#      # Look up embeddings for inputs.
#      with tf.name_scope('embeddings'):
#        embeddings = tf.Variable(
#            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
#        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
#
#      # Construct the variables for the NCE loss
#      with tf.name_scope('weights'):
#        nce_weights = tf.Variable(
#            tf.truncated_normal([vocabulary_size, embedding_size],
#                                stddev=1.0 / math.sqrt(embedding_size)))
#      with tf.name_scope('biases'):
#        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    #   测试默认是否运行 gpu
    gpuconfig = tf.ConfigProto(allow_soft_placement=True)
    gpuconfig.gpu_options.allow_growth = True
    
      # Look up embeddings for inputs.
      # 随机生成embeddings (50000,128),embed为从中提取labels 的128 index对应结果
      # embed:(128,128) embeddings 跑一次变一次，如果embed在embeddings 后返回， 返回的结果跟返回embeddings 对应index 不一致
    with tf.name_scope('embeddings'):
      embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)

      # Construct the variables for the NCE loss
    with tf.name_scope('weights'):
      nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
    with tf.name_scope('biases'):
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    with tf.name_scope('loss'):
      logger.debug('nce_weights.shape : {};nce_biases.shape : {}; embed.shape : {}'.format(nce_weights.shape,nce_biases.shape,embed.shape))
#      print('nce_weights.shape : {};nce_biases.shape : {};train_labels.shape : {}; embed.shape : {}'.format(nce_weights.shape,nce_biases.shape,train_labels.shape,embed.shape))
      loss = tf.reduce_mean(
          tf.nn.nce_loss(
              weights=nce_weights,
              biases=nce_biases,
              labels=train_labels,
              inputs=embed,
              num_sampled=num_sampled,
              num_classes=vocabulary_size))

    # Add the loss value as a scalar to summary.
    #   一般在画loss曲线和accuary曲线时会用到这个函数。
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all
    # embeddings.
    #   norm:(50000,1) : embedding (50000,128) tf.square：每个元素平方，tf.reduce_sum：行求和生成[[sum_l1],[sum_l2],……,[sum_l50000]],最后取平方根
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    #   相当于标准化 ： x1j/(sumj=(1,128)(x1j^2))^0.5
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
    #   valid_embeddings(16,128) 成 normalized_embeddings (50000,128).T 
        #   即：(16,128) * (128,50000)
        # 相似度 ： cos = a*b/|a|*|b| = a/|a| * b/|b| 
#                      = valid_embeddings * normalized_embeddings
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

  # Step 5: Begin training.
  
  
  num_steps = 100001

  with tf.Session(graph=graph,config = gpuconfig) as session:
    logger.debug('begin Session  ')
      
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(log_dir, session.graph)

    # We must initialize all variables before we use them.
    init.run()
    logger.debug('Initialized')
    logger.debug('--------- begin n_steps -------')
    average_loss = 0
    for step in range(num_steps):        
      if step < 10 :
          logger.debug('step : {} , average_loss :{}'.format(step,average_loss))
          batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
                                                      skip_window)
          try:
              logger.debug('1 batch_inputs : {}'.format(batch_inputs))
              logger.debug('1 batch_labels : {}'.format(batch_labels))
          except:
              logger.debug('batch_inputs : {}'.format(session.run(batch_inputs)))
              logger.debug('batch_labels : {}'.format(session.run(batch_labels)))
          
          feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
#          logger.debug('feed_dict : {}'.format(feed_dict))
          logger.debug('******* nce_loss :********* ')
          logger.debug('embeddings : {} ; '.format(embeddings.eval()))
          logger.debug('imputs => embed : {} '.format(session.run(embed,feed_dict = {train_inputs : batch_inputs})))
          logger.debug('weights => nce_weights : {}  '.format(nce_weights.eval()))
          logger.debug('biases => nce_biases : {}  '.format(nce_biases.eval()))
          logger.debug('num_sampled : {} ; num_classes => vocabulary_size: {}'.format(num_sampled,vocabulary_size))
          try:
              logger.debug('norm : {} ; '.format(norm.eval()))
              logger.debug('normalized_embeddings : {} ;\n normalized_embeddings.shape :{} '.format(normalized_embeddings.eval(),normalized_embeddings.shape))
              logger.debug('valid_dataset : {} ; '.format(valid_dataset.eval()))
              logger.debug('similarity : {} ;\n similarity.shape :{} '.format(similarity.eval(),similarity.shape))          
          except Exception as e:
              logger.error('can not get values of simlarity and its detail cause : {} '.format(e))
          
          # Define metadata variable.
          run_metadata = tf.RunMetadata()
    
          # We perform one update step by evaluating the optimizer op (including it
          # in the list of returned values for session.run()
          # Also, evaluate the merged op to get all summaries from the returned
          # "summary" variable. Feed metadata variable to session for visualizing
          # the graph in TensorBoard.
          _, summary, loss_val = session.run([optimizer, merged, loss],
                                             feed_dict=feed_dict,
                                             run_metadata=run_metadata)
          average_loss += loss_val

          logger.debug('[optimizer, merged, loss] : {}'.format(session.run([optimizer, merged, loss],
                                             feed_dict=feed_dict,
                                             run_metadata=run_metadata)))
          logger.debug('average_loss : {}'.format(average_loss))
      else:
          batch_inputs, batch_labels = generate_batch_free(batch_size, num_skips,
                                                      skip_window)
        
          feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
    
          # Define metadata variable.
          run_metadata = tf.RunMetadata()
    
          # We perform one update step by evaluating the optimizer op (including it
          # in the list of returned values for session.run()
          # Also, evaluate the merged op to get all summaries from the returned
          # "summary" variable. Feed metadata variable to session for visualizing
          # the graph in TensorBoard.
          _, summary, loss_val = session.run([optimizer, merged, loss],
                                             feed_dict=feed_dict,
                                             run_metadata=run_metadata)
          average_loss += loss_val
         
          
      # Add returned summaries to writer in each step.
      writer.add_summary(summary, step)
      # Add metadata to visualize the graph for the last run.
      if step == (num_steps - 1):
        writer.add_run_metadata(run_metadata, 'step%d' % step)

      if step % 2000 == 0:
        if step > 0:
          average_loss /= 2000
        # The average loss is an estimate of the loss over the last 2000
        # batches.
        print('Average loss at step ', step, ': ', average_loss)
        logger.debug('Average loss at step  : {} ;average_loss : {}'.format(step,average_loss))
        average_loss = 0

      # Note that this is expensive (~20% slowdown if computed every 500 steps)
      if step % 10000 == 0:
                      
        logger.debug('embeddings : {} ; '.format(embeddings.eval()))
        logger.debug('norm : {} ; '.format(norm.eval()))
        logger.debug('normalized_embeddings : {} ; '.format(normalized_embeddings.eval()))
        logger.debug('valid_dataset : {} ; '.format(valid_dataset.eval()))
#        logger.debug('similarity : {} ; '.format(similarity.eval()))
        sim = similarity.eval()
        logger.debug('sim : {} ; '.format(sim))
        logger.debug('{} going to range valid_size : {} times'.format('-'*30,valid_size))
        for i in range(valid_size):
          logger.debug(' ----------------- i : {} turn '.format(i))
          valid_word = reverse_dictionary[valid_examples[i]]
          top_k = 8  # number of nearest neighbors
          # argsort 返回从大到小的索引值
          nearest = (-sim[i, :]).argsort()[1:top_k + 1]
          log_str = 'Nearest to %s:' % valid_word
          logger.debug('valid_examples[ {} ] : {}; \n valid_word : {} \n'.format(i,valid_examples[i],valid_word))
          logger.debug('-sim[{}, :] : {}'.format(i,-sim[i, :]))
          logger.debug('(-sim[ {} , :]).argsort() : {}'.format(i,(-sim[i, :]).argsort()))
          logger.debug('nearest = (-sim[ {}, :]).argsort()[1:{} ] : {}'.format(i,top_k + 1,nearest))
          logger.debug('log_str : {}'.format(log_str))
          
          for k in range(top_k):
            logger.debug('--------------------top_k k: {} turn'.format(k))
            close_word = reverse_dictionary[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
            logger.debug('close_word : {}; \n log_str : {}'.format(close_word,log_str))
          print(log_str)
          
    final_embeddings = normalized_embeddings.eval()
    logger.debug('final_embeddings :  {}  '.format(final_embeddings))
    # Write corresponding labels for the embeddings.
    with open(log_dir + '/metadata.tsv', 'w') as f:
      for i in range(vocabulary_size):
        f.write(reverse_dictionary[i] + '\n')

    # Save the model for checkpoints.
    saver.save(session, os.path.join(log_dir, 'model.ckpt'))

    # Create a configuration for visualizing embeddings with the labels in
    # TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

  writer.close()

  # Step 6: Visualize the embeddings.
  logger.debug('Step 6: Visualize the embeddings ')
  # pylint: disable=missing-docstring
  # Function to draw visualization of distance between embeddings.
  def plot_with_labels(low_dim_embs, labels, filename):
    logger.debug('-----  enter plot_with_labels -----')
    logger.debug('low_dim_embs.shape : {} , len(labels) : {} ; filename : {}'.format(low_dim_embs.shape,len(labels),filename))
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
      x, y = low_dim_embs[i, :]
      plt.scatter(x, y)
      plt.annotate(
          label,
          xy=(x, y),
          xytext=(5, 2),
          textcoords='offset points',
          ha='right',
          va='bottom')

    plt.savefig(filename)
    logger.debug('-----  end plot_with_labels -----')
  try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    #   降维
    logger.debug('before tsne.fit_transform  data.shape : {} '.format(final_embeddings[:plot_only, :].shape))
    logger.debug('data[0] : \n {}'.format(final_embeddings[:1, :]))
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    logger.debug('after tsne.fit_transform  data.shape : {} '.format(low_dim_embs.shape))
    logger.debug('data[0] : \n {}'.format(low_dim_embs[0]))

    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(),
                                                        'tsne.png'))

  except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)
    logger.debug('tsne error : {}'.format(ex))


# All functionality is run after tf.compat.v1.app.run() (b/122547914). This
# could be split up but the methods are laid sequentially with their usage for
# clarity.
def main(unused_argv):
  # Give a folder path as an argument with '--log_dir' to save
  # TensorBoard summaries. Default is a log folder in current directory.
  current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
  logger.debug('current_path : {}'.format(current_path))
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(current_path, 'log'),
      help='The log directory for TensorBoard summaries.')
  flags, unused_flags = parser.parse_known_args()
  word2vec_basic(flags.log_dir)


if __name__ == '__main__':
  tf.app.run()
    
    #如果你的代码中的入口函数叫main()，则你就可以把入口写成tf.app.run()
    #如果你的代码中的入口函数不叫main()，而是一个其他名字的函数，如test()，则你应该这样写入口tf.app.run(test)

#    python word2vec_basic.py --log_dir D:\myGit\Algorithm\tensorflow_example\word2vec_basic\model
  
  
