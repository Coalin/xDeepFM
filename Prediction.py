#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Zhou, Jing'

import tensorflow as tf
import Config
from tools import _get_data, _get_conf, get_label, auc_score
import pandas as pd


meta_path = './checkpoint_dir/MyModel.meta'
model_path = './checkpoint_dir/MyModel'


def _init_placeholder(graph):
    ph = {}
    ph['label'] = graph.get_tensor_by_name('label:0')
    train_phase = graph.get_tensor_by_name("train_phase:0")
    ph['value'] = graph.get_tensor_by_name(name='value:0')
    ph['single_index'] = graph.get_tensor_by_name(name='single_index:0')
    ph['numerical_index'] = graph.get_tensor_by_name('numerical_index:0')
    for s in Config.multi_features:
        ph['multi_index_%s' % s] = graph.get_tensor_by_name('multi_index_%s:0' % s)
        ph['multi_value_%s' % s] = graph.get_tensor_by_name('multi_value_%s:0' % s)
    if not Config.use_numerical_embedding:
        ph['numerical_value'] = graph.get_tensor_by_name('numerical_value:0')
    return ph, train_phase


def _get_batch(data, idx, single_size=None, numerical_size=None, multi_size=None):
    if idx == -1:
        batch_data = data
    elif (idx + 1) * Config.batch_size <= len(data):
        batch_data = data[idx*Config.batch_size:(idx+1)*Config.batch_size]
    else:
        batch_data = data[idx*Config.batch_size:]
    final_label = []
    final_single_index = []
    final_numerical_value = []
    final_numerical_index = []
    final_multi_sparse_index = []
    final_multi_sparse_value = []
    final_value = []
    for idx, line in enumerate(batch_data):
        line_index = []
        line_value = []
        line_numerical_value = []
        line_data = line.split(',')
        final_label.append(int(line_data[0]))

        if single_size:
            for i in range(1, 1 + single_size):
                single_pair = line_data[i].split(':')
                line_index.append(int(single_pair[0]))
                line_value.append(float(single_pair[1]))
        final_single_index.append(line_index)
        line_index = []

        if single_size + numerical_size:
            for i in range(1 + single_size, 1 + single_size + numerical_size):
                single_pair = line_data[i].split(':')
                if not Config.use_numerical_embedding:
                    line_numerical_value.append(float(single_pair[1]))
                if float(single_pair[1]) == 0:
                    line_index.append(int(9999))
                    line_value.append(float(1))
                else:
                    line_index.append(int(single_pair[0]))
                    line_value.append(float(single_pair[1]))
        final_numerical_value.append(line_numerical_value)
        final_numerical_index.append(line_index)
        line_index = []
        total_length = 1 + single_size + numerical_size + multi_size

        if multi_size:
            for i in range(1 + single_size + numerical_size, total_length):
                single_pair = line_data[i].split(':')
                _multi = [int(x) for x in single_pair[0].split('-')]
                line_index.append(_multi)
                for v in _multi:
                    final_multi_sparse_index.append([idx, idx])
                    final_multi_sparse_value.append(v)
                line_value.append(float(single_pair[1]))
        final_value.append(line_value)
    return [final_label, final_single_index, final_numerical_index, final_numerical_value, final_multi_sparse_index, final_multi_sparse_value, final_value]


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(meta_path)
    new_saver.restore(sess, model_path)
    # init = tf.global_variables_initializer()
    # sess.run(init)

    graph = tf.get_default_graph()
    ph, train_phase = _init_placeholder(graph)
    print(ph)

    prediction = tf.get_collection('pred_network')[0]
    # loss = tf.get_collection('loss')[0]
    loss = graph.get_tensor_by_name('loss:0')

    total_emb, single_size, numerical_size, multi_size = _get_conf()
    field_size =single_size + numerical_size + multi_size
    embedding_length = field_size * Config.embedding_size

    test = _get_data(Config.test_save_file)
    test = test[:30000]
    test_batch = _get_batch(test, -1, single_size=single_size, numerical_size=numerical_size, multi_size=multi_size)
    test_label = get_label(test_batch[0], 2)

    test_dict = {
        ph['single_index']: test_batch[1],
        ph['numerical_index']: test_batch[2],
        ph['numerical_value']: test_batch[3],
        ph['value']: test_batch[-1],
        ph['label']: test_label,
        train_phase: False
    }
    if Config.multi_features:
        for idx, s in enumerate(Config.multi_features):
            test_dict[ph['multi_index_%s' % s]] = test_batch[4]
            test_dict[ph['multi_value_%s' % s]] = test_batch[5]

    prediction, loss = sess.run((prediction, loss), feed_dict=test_dict)
    print(loss)
    print(prediction)
    auc_for_test = auc_score(prediction, get_label(test_batch[0], 2), 2)
    print(auc_for_test)
    res = []
    test_out_ = [x[1] for x in prediction]
    print(test_out_)
    # res.append([test_out_])
    # print(res)
    res_df = pd.DataFrame(test_out_)
    res_df.to_csv("./ex_data/results.csv")

print('Its OK')



