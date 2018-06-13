# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: vec_transfer_classifier.py
@time: 18-6-13 下午9:25
@description: 药物特征（flatten为一维）经过矩阵变换后再拼接在一起输入到conv分类器中
"""
import sys
sys.path.extend(['/home/cdy/ykq/DDISuccess', '/home/cdy/ykq/DDISuccess/Base'])

from Base.native_classifier import lenet5
from utils import load_dataset_split, permute_dataset
import tensorflow as tf
import prettytensor as pt

batch_size = 100
epochs = 800  # Number of epochs through the full dataset
learning_rate = 3e-4
num_lab = 28800
dim_y = 2
latent_dim = 500
classify_dataset_path = "/home/cdy/ykq/DDISuccess/Base/BaseDataset/TestDataset"

train_data, train_labels, _, __, test_data, test_labels = load_dataset_split(classify_dataset_path, num_lab//2, 3)
dim_x = train_data.shape[1]
feature_dim = dim_x / 2
train_batch_num = train_data.shape[0] / batch_size
test_batch_num = test_data.shape[0] / batch_size
test_head_data = test_data[:, 0:feature_dim]
test_tail_data = test_data[:, feature_dim:dim_x]


head_data_placeholder = tf.placeholder(tf.float32, [batch_size, feature_dim])
tail_data_placeholder = tf.placeholder(tf.float32, [batch_size, feature_dim])
labels_placeholder    = tf.placeholder(tf.float32, [batch_size, dim_y])

trans_matrix = tf.Variable(tf.float32, [batch_size, feature_dim, latent_dim])

head_trans = tf.matmul(head_data_placeholder, trans_matrix)
tail_trans = tf.matmul(tail_data_placeholder, trans_matrix)
input_data = tf.reshape(tf.concat([head_trans, tail_trans], 1), [batch_size, dim_x, 1])

result = lenet5(input_data, labels_placeholder, dim_y)
accuracy = result.softmax.evaluate_classifier(labels_placeholder, phase=pt.Phase.test)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = pt.apply_optimizer(optimizer, losses=[result.loss])

runner = pt.train.Runner()
best_accuracy = 0
best_epoch = 0
with tf.Session() as sess:
    for epoch in epochs:
        train_data, train_labels = permute_dataset((train_data, train_labels))
        head_train_data = train_data[:, 0:feature_dim]
        tail_train_data = train_data[:, feature_dim:dim_x]
        runner.train_model(train_op, result.loss, train_batch_num,
                           feed_vars=(head_data_placeholder, tail_data_placeholder, labels_placeholder),
                           feed_data=(pt.train.feed_numpy(batch_size, head_train_data, tail_train_data, train_labels)))
        classification_accuracy = runner.evaluate_model(accuracy, test_batch_num,
                                                        feed_vars=(head_data_placeholder, tail_data_placeholder, labels_placeholder),
                                                        feed_data=(pt.train.feed_numpy(batch_size, test_head_data, test_tail_data, test_labels)))
        if best_accuracy < classification_accuracy[0]:
            best_accuracy = classification_accuracy
            best_epoch = epoch
        print('Accuracy after %d epoch %g%%' % (epoch + 1, classification_accuracy[0] * 100))
    print('Train size is {}.Best accuracy is {}%% at {} epoch.'.format(num_lab, best_accuracy, best_epoch))
    print('==================================')

# start with 65.667