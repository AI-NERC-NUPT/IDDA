########################################################################################
#   Unsupervised Domain Adaptation for SVHN-->MNIST
#   1. Batch-normalization is only employed at the Conv-3 layer in the shared network F
#   2. Drop-out is disabled in the 0-th phase, but used in the subsequent phases
#   3. Batch size for training Ft, F is set as 128, for training Fs, F is set as 64
########################################################################################
import tensorflow as tf
import os
import numpy as np
from utils import return_mnist, return_svhn, judge_init_func, judge_idda_func, pick_samples, weight_variable, \
    bias_variable, max_pool_3x3, conv2d, batch_norm_conv, batch_norm_fc, batch_generator
flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.025, "value of learnin rage")
FLAGS = flags.FLAGS
N_CLASS = 10

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #gpu

#insert your path to dataset
path_svhn_train = 'Yourpath/data/svhn/train_32x32.mat'
path_svhn_test = 'Yourpath/data/svhn/test_32x32.mat'
path_mnist_train = 'Yourpath/data/mnist/train_mnist_32x32.npy'
path_mnist_test = 'Yourpath/data/mnist/test_mnist_32x32.npy'
print('data loading...')

data_s_im, data_s_label, data_s_im_test, data_s_label_test = return_svhn(path_svhn_train, path_svhn_test)
data_t_im, data_t_label, data_t_im_test, data_t_label_test = return_mnist(path_mnist_train, path_mnist_test)
print('load finished')
# Compute pixel mean for normalizing data
pixel_mean = np.vstack([data_t_im, data_s_im]).mean((0, 1, 2))
num_test = 500
batch_size = 128

class Model(object):

    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.X = tf.placeholder(tf.uint8, [None, 32, 32, 3])
        self.y = tf.placeholder(tf.float32, [None, N_CLASS])
        self.train = tf.placeholder(tf.bool, [])
        self.keep_prob = tf.placeholder(tf.float32)
        self.classify_labels = self.y

        X_input = (tf.cast(self.X, tf.float32) - pixel_mean) / 255.
        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            W_conv0 = weight_variable([5, 5, 3, 64], stddev=0.01, name='W_conv0')
            b_conv0 = bias_variable([64], init=0.01, name='b_conv0')
            h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)
            h_pool0 = max_pool_3x3(h_conv0)

            W_conv1 = weight_variable([5, 5, 64, 64], stddev=0.01, name='W_conv1')
            b_conv1 = bias_variable([64], init=0.01, name='b_conv1')
            h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
            h_pool1 = max_pool_3x3(h_conv1)

            W_conv2 = weight_variable([5, 5, 64, 128], stddev=0.01, name='W_conv2')
            b_conv2 = bias_variable([128], init=0.01, name='b_conv1')
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_conv2 = batch_norm_conv(h_conv2, 128)

            h_fc1_drop = tf.nn.dropout(h_conv2, self.keep_prob)
            h_fc1_drop = tf.reshape(h_fc1_drop, [-1, 8192])

            W_fc_0 = weight_variable([8192, 3072], stddev=0.01, name='W_fc0')
            b_fc_0 = bias_variable([3072], init=0.01, name='b_fc0')
            h_fc_0 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc_0) + b_fc_0)

            self.feature = tf.nn.dropout(h_fc_0, self.keep_prob)


        with tf.variable_scope('label_predictor_source'):
            W_fc0 = weight_variable([3072, 2048], stddev=0.01, name='W_fc0')
            b_fc0 = bias_variable([2048], init=0.01, name='b_fc0')
            h_fc0 = tf.nn.relu(batch_norm_fc(tf.matmul(self.feature, W_fc0) + b_fc0, 2048))
            h_fc0 = tf.nn.dropout(h_fc0, self.keep_prob)

            W_fc1 = weight_variable([2048, N_CLASS], stddev=0.01, name='W_fc1')
            b_fc1 = bias_variable([N_CLASS], init=0.01, name='b_fc1')
            logits = tf.matmul(h_fc0, W_fc1) + b_fc1

            classify_logits = logits
            self.pred_s = tf.nn.softmax(classify_logits)
            self.pred_loss_s = tf.nn.softmax_cross_entropy_with_logits(logits=classify_logits,
                                                                       labels=self.classify_labels)

        with tf.variable_scope('label_predictor_target'):
            W_fc0_t = weight_variable([3072, 2048], stddev=0.01, name='W_fc0_t')
            b_fc0_t = bias_variable([2048], init=0.01, name='b_fc0_t')
            h_fc0_t = tf.nn.relu(tf.matmul(self.feature, W_fc0_t) + b_fc0_t)
            h_fc0_t = tf.nn.dropout(h_fc0_t, self.keep_prob)

            W_fc1_t = weight_variable([2048, 10], stddev=0.01, name='W_fc1_t')
            b_fc1_t = bias_variable([10], init=0.01, name='b_fc1_t')
            logits_t = tf.matmul(h_fc0_t, W_fc1_t) + b_fc1_t

            classify_logits = logits_t

            self.pred_t = tf.nn.softmax(classify_logits)
            self.pred_loss_t = tf.nn.softmax_cross_entropy_with_logits(logits=classify_logits,
                                                                       labels=self.classify_labels)


graph = tf.get_default_graph()
with graph.as_default():
    model = Model()
    learning_rate = tf.placeholder(tf.float32, [])
    pred_loss_source = tf.reduce_mean(model.pred_loss_s)
    pred_loss_target = tf.reduce_mean(model.pred_loss_t)

    source_loss = pred_loss_source
    target_loss = pred_loss_target
    total_loss = pred_loss_source + pred_loss_target  #may be mistaken

    ## 1. Optimize  source net (F->Fs) with source_loss:= pred_loss_source
    source_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(source_loss)

    ## 2. Optimize the target net Ft-net (F->Ft) with target_loss = pred_loss_target
    target_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(target_loss)

    # Evaluation
    correct_label_pred_s = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred_s, 1))
    correct_label_pred_t = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred_t, 1))

    label_acc_t = tf.reduce_mean(tf.cast(correct_label_pred_t, tf.float32))
    label_acc_s = tf.reduce_mean(tf.cast(correct_label_pred_s, tf.float32))

    # Investigation
    num_correct_s = tf.reduce_sum(tf.cast(correct_label_pred_s, tf.float32)) + 0.01     #+0.01 to avoid zero
    net_s_correct_loss = 1/num_correct_s * tf.reduce_sum(model.pred_loss_s * tf.cast(correct_label_pred_s, tf.float32))

# Params
num_steps = 3000
num_phases = 30

def train_and_evaluate(graph, model, verbose=True):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()
        # Batch generators
        for t in range(num_phases):
            print('phase:%d' % (t))  #In each phase, num_steps = 3000 batches are employed for training?
            label_target = np.zeros((data_t_im.shape[0], N_CLASS))
            if t == 0:
                gen_source_only_batch = batch_generator(
                    [data_s_im, data_s_label], batch_size//2)  #shuffle is employed for dataset with default
            else:
                ####-------source_train  always starts with data_s, which is not satisfied for unsup.---
                source_train = data_s_im
                source_label = data_s_label

                source_train = np.r_[source_train, new_data]
                source_label = np.r_[source_label, new_label]

                gen_source_batch = batch_generator(
                    [source_train, source_label], batch_size//2)
                gen_new_batch = batch_generator(
                    [new_data, new_label], batch_size)

            # Training loop
            for i in range(num_steps):
                lr = FLAGS.learning_rate
                dropout = 0.5
                # Training step
                if t == 0:  #phase 0
                    X0, y0 = next(gen_source_only_batch) #source only
                    _, _, batch_loss, p_loss_s, p_acc_s = \
                        sess.run([target_train_op, source_train_op, total_loss, pred_loss_source,
                                  label_acc_s],
                                 feed_dict={model.X: X0, model.y: y0,
                                            model.train: False, learning_rate: lr, model.keep_prob: 1})
                    if verbose and i % 5000 == 0:
                        print('loss: %f  loss_s: %f  acc_s: %f' % \
                              (batch_loss, p_loss_s, p_acc_s))

                if t >= 1:  #phase 1, 2, ...
                    X0, y0 = next(gen_source_batch)  #souce only + new_data(target)
                    _, batch_loss, p_loss_s, p_acc_s = \
                        sess.run([source_train_op, total_loss, pred_loss_source,
                                  label_acc_s],
                                 feed_dict={model.X: X0, model.y: y0, model.train: False, learning_rate: lr,
                                            model.keep_prob: dropout})

                    X1, y1 = next(gen_new_batch) #new_data only (target)
                    _, p_acc_t = \
                        sess.run([target_train_op, label_acc_t],
                                 feed_dict={model.X: X1, model.y: y1, model.train: False, learning_rate: lr,
                                            model.keep_prob: dropout})

                    if verbose and i % 5000 == 0:
                        print('loss: %f  loss_s: %f  acc_s: %f acc_t: %f' % \
                              (batch_loss, p_loss_s, p_acc_s, p_acc_t))
           # Attach Pseudo Label
            step = 0
            preds_stack = np.zeros((0, N_CLASS))
            predt_stack = np.zeros((0, N_CLASS))
            stack_num = min(data_t_im.shape[0] // batch_size, 100 * (t + 1))  #?????
            # Shuffle pseudo labeled candidates
            perm = np.random.permutation(data_t_im.shape[0])
            gen_target_batch = batch_generator(
                [data_t_im[perm, :], label_target], batch_size, shuffle=False)
            while step < stack_num:  #stack_num = target_set_size/batch_size
                if t == 0:
                    X1, y1 = next(gen_target_batch)
                    pred_s = sess.run(model.pred_s,
                                      feed_dict={model.X: X1,
                                                 model.y: y1,
                                                 model.train: False,
                                                 model.keep_prob: 1})
                    preds_stack = np.r_[preds_stack, pred_s]
                    step += 1
                else:
                    X1, y1 = next(gen_target_batch)

                    pred_s, pred_t = sess.run([model.pred_s, model.pred_t],
                                              feed_dict={model.X: X1,
                                                         model.y: y1,
                                                         model.train: False,
                                                         model.keep_prob: 1})
                    preds_stack = np.r_[preds_stack, pred_s]
                    predt_stack = np.r_[predt_stack, pred_t]
                    step += 1
            if t == 0:
                cand = data_t_im[perm, :]
                canl = data_t_label[perm,:]
                rate = max(int((t + 1) / 20.0 * preds_stack.shape[0]), 2000)  # k/20 * n?: number of newly-labeled target samples
                new_data, new_label, new_ind = judge_idda_func(cand,
                                                             preds_stack[:rate, :],
                                                             lower = 0.09,
                                                             num_class=N_CLASS)
                correct_new_labels = tf.equal(tf.argmax(new_label, 1), tf.argmax(canl[new_ind,:],1))
                label_correct_ones = tf.reduce_sum(tf.cast(correct_new_labels, tf.int32))
                label_errors = new_data.shape[0] - label_correct_ones

            lowerValue = 0.12 * pow(2,-t) + 0.005
            if t != 0:
                cand = data_t_im[perm, :]
                canl = data_t_label[perm, :]
                rate = max(int((t + 1) / 1.0 * preds_stack.shape[0]), 60000)
                new_data, new_label, new_ind = judge_init_func(cand,
                                                             preds_stack[:rate, :],
                                                             lower = 0.01,     #0.06
                                                             num_sel = 1024,
                                                             num_class=N_CLASS)
                
                correct_new_labels = tf.equal(tf.argmax(new_label, 1), tf.argmax(canl[new_ind, :], 1))
                label_correct_ones = tf.reduce_sum(tf.cast(correct_new_labels, tf.int32))
                label_errors = new_data.shape[0] - label_correct_ones

            print('new_data_size: %d'% (new_data.shape[0]))  #print('phase:%d' % (t))
            print('new_label_errors: %d'% (sess.run(label_errors)))

            # Evaluation
            gen_source_batch = batch_generator(
                [data_s_im_test, data_s_label_test], batch_size, test=True)
            gen_target_batch = batch_generator(
                [data_t_im_test, data_t_label_test], batch_size, test=True)
            num_iter = int(data_t_im_test.shape[0] / batch_size) + 1
            step = 0
            total_source = 0
            total_target = 0
            total_acc_s = 0
            aver_net_s_loss = 0
            aver_net_s_correct_loss = 0
            size_t = 0
            size_s = 0
            while step < num_iter:
                X0, y0 = next(gen_source_batch)
                X1, y1 = next(gen_target_batch)
                source_acc = sess.run(label_acc_s,
                                      feed_dict={model.X: X0, model.y: y0,
                                                 model.train: False, model.keep_prob: 1})
                net_s_correct_loss_per_batch, net_s_loss_per_batch, target_acc, t_acc_s = \
                    sess.run([net_s_correct_loss, source_loss, label_acc_t, label_acc_s],
                             feed_dict={model.X: X1, model.y: y1, model.train: False,
                                        model.keep_prob: 1})
                aver_net_s_loss += net_s_loss_per_batch
                aver_net_s_correct_loss += net_s_correct_loss_per_batch
                total_source += source_acc * len(X0)
                total_target += target_acc * len(X1)
                total_acc_s += t_acc_s * len(X1)
                size_t += len(X1)
                size_s += len(X0)
                step += 1

            # print('new_label_errors: %d'% (label_errors))
            print('Eval source-net-loss with target domain', aver_net_s_loss / num_iter)
            print('Eval source-net-correct-loss with target domain', aver_net_s_correct_loss / num_iter)
            print('lower threshold', lowerValue)  # 0.08 * pow(1.2, -t) + 0.04

            print('train target', total_target / size_t, total_acc_s / size_t, total_source / size_s)
    return total_source / size_s, total_target / size_t, total_acc_s / size_t

print('\nTraining Start')
all_source = 0
all_target = 0
all_t_snet = 0

num_experiments = 10
acc_t_snet=[]
acc_t_tnet=[]
for i in range(num_experiments):
    source_acc, target_acc, t_acc_s = train_and_evaluate(graph, model)
    acc_t_snet.append(t_acc_s)
    acc_t_tnet.append(target_acc)
    all_source += source_acc
    all_target += target_acc
    all_t_snet += t_acc_s
    print('Source accuracy:', source_acc)
    print('Target accuracy (Target Classifier):', target_acc)
    print('Target accuracy (Source Classifier):', t_acc_s)

print('Avg Source accuracy: %f'% (all_source / num_experiments))

print('Per Target accuracy (Target Classifier):', acc_t_tnet)
print('Per Target accuracy (Source Classifier):', acc_t_snet)

print('Avg Target accuracy-tnet: %f'% (all_target / num_experiments))
print('Avg Target accuracy-snet: %f'% (all_t_snet / num_experiments))
