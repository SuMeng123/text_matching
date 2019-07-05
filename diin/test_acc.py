import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from diin.graph import Graph
import tensorflow as tf
from diin import args
from utils.load_data import load_char_word_dynamic_data, sumeng_load_char_word_dynamic_data
import numpy as np
import pickle
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# p_c_index_holder = tf.placeholder(name='p_index', shape=(None, args.max_char_len), dtype=tf.int32)
# h_c_index_holder = tf.placeholder(name='h_index', shape=(None, args.max_char_len), dtype=tf.int32)
# p_w_index_holder = tf.placeholder(name='p_vec', shape=(None, args.max_word_len), dtype=tf.int32)
# h_w_index_holder = tf.placeholder(name='h_vec', shape=(None, args.max_word_len), dtype=tf.int32)
# label_holder = tf.placeholder(name='label', shape=(None,), dtype=tf.int32)
#
# dataset = tf.data.Dataset.from_tensor_slices(
#     (p_c_index_holder, h_c_index_holder, p_w_index_holder, h_w_index_holder, label_holder))
# dataset = dataset.batch(args.batch_size).repeat(args.epochs)
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()

with open('../output/diin/w2v.vec', 'rb')as file:
    embedding = pickle.load(file)
model = Graph(word_embedding=embedding)
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    # sess.run(iterator.initializer, feed_dict={p_c_index_holder: p_c_index,
    #                                           h_c_index_holder: h_c_index,
    #                                           p_w_index_holder: p_w_index,
    #                                           h_w_index_holder: h_w_index,
    #                                           label_holder: label})
    saver.restore(sess, "../output/diin/diin_3.ckpt")
    # steps = int(len(label) / args.batch_size)
    loss_all = []
    acc_all = []
    # for step in range(steps):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    try:
        df = pd.read_csv('../input/test.csv')
        p = df['sentence1'].values[:]
        h = df['sentence2'].values[:]
        labels = df['label'].values[:]
        for index,label in enumerate(labels):
            sentence1 = p[index]
            sentence2 = h[index]
            p_c_index, h_c_index, p_w_index, h_w_index, _ = sumeng_load_char_word_dynamic_data(sentence1, sentence2)
            # p_index_batch, h_index_batch, p_vec_batch, h_vec_batch, label_batch = sess.run(next_element)
            predict = sess.run([ model.predict],
                                             feed_dict={model.p_c: p_c_index,
                                                        model.h_c: h_c_index,
                                                        model.p_w: p_w_index,
                                                        model.h_w: h_w_index,
                                                        model.y: [label],
                                                        model.keep_prob: args.keep_prob})
            if label == 1 and predict[0] ==1:
                TP += 1
            elif label == 1 and predict[0] ==0:
                FN += 1
            elif label == 0 and predict[0] == 0:
                TN += 1
            elif label == 0 and predict[0] ==1:
                FP += 1
            else:
                pass
        print("precision:",TP/(TP+FP))
        print("Recall:",TP/(TP+FN))

    except tf.errors.OutOfRangeError:
        print('报错')

if __name__ =="__main__":
    print("hello")