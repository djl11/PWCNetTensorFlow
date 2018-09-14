import tensorflow as tf
import os
import math
import numpy as np
import custom_ops.native as native_ops
import custom_ops.compiled as compiled_ops

class Dims():
    def __init__(self):
        self.input_vector = 0
        self.input_image = [384,448,3]
        self.output = [384,448,2]

class Network():

    def __init__(self, data_loader, dirs):

        self.data_loader = data_loader
        self.chkpt_dir = dirs.chkpt_dir
        self.log_dir = dirs.log_dir
        self.log_dir_training = dirs.log_dir + 'training/'
        self.log_dir_validation = dirs.log_dir + 'validation/'

        self.dims = Dims()

        self.global_step = tf.placeholder(tf.int32)

        self.window_size = 11
        self.max_dist = int(math.floor(self.window_size/2))

        self.batch_size = 8

        self.initial_learning_rate = 2e-4
        self.learning_decrement_rate = 0.2e6
        self.learning_decrement = 0.5
        self.min_learning_rate = 6.25e-6
        self.max_learning_rate = 1e-4

        self.total_iterations = 1e6

        self.validation_ratio = 0.1
        self.num_training_examples = int(round(self.data_loader.total_num_examples * (1 - self.validation_ratio)))
        self.num_validation_examples = self.data_loader.total_num_examples - self.num_training_examples
        self.total_num_examples = self.num_training_examples + self.num_validation_examples

        self.define_placeholders()

    @staticmethod
    def directory():
        return os.path.dirname(os.path.realpath(__file__))


    # Operations #
    #------------#

    def warp(self, ct, w):
        return native_ops.image_warp(ct, w)

    def cost_volume(self, cwt, ctm1):
        return compiled_ops.correlation(cwt, ctm1, pad=4, kernel_size=1, max_displacement=4, stride_1=1, stride_2=1)


    # Inputs #
    #--------#

    def define_placeholders(self):

        self.data_type = tf.placeholder(tf.string)
        self.vis_mode = tf.placeholder(tf.bool)
        self.step_placeholder = tf.placeholder(tf.int32)

    def define_data_loader(self):

        self.loaded_data = list(self.data_loader.next_element)
        self.loaded_images = tf.cast(self.loaded_data[0],tf.float32)
        self.loaded_flow_images = tf.cast(self.loaded_data[1][:,0:1],tf.float32)/20

    def define_network_inputs(self):

        # rgb images
        self.stacked_images = tf.reshape(self.loaded_images, (-1, 384, 512, 3))

        # flow images
        self.appended_flow_images = tf.concat((self.loaded_flow_images, tf.zeros((self.batch_size,1, 384, 512, 2))), 1)
        self.stacked_flow_images = tf.reshape(self.appended_flow_images, (-1, 384, 512, 2))

        # combined images
        self.stacked_combined_images = tf.concat((self.stacked_images, self.stacked_flow_images),-1)
        self.cropped_combined_images = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img,
                                                    0, tf.random_uniform((), 0, 512 - 448, tf.int32),
                                                    384, 448), self.stacked_combined_images)

        self.unstacked_combined_images = tf.reshape(self.cropped_combined_images, (self.batch_size, 2, self.dims.input_image[0],
                                                    self.dims.input_image[1], 5))

        # rgb images
        self.unstacked_images = self.unstacked_combined_images[:,:,:,:,0:3]
        self.x_images = tf.transpose(self.unstacked_images, [0,1,4,2,3])

        # flow images
        self.x_flow_images = self.unstacked_combined_images[:,0,:,:,3:5]

    def define_network_targets(self):

        # ground truth outputs
        self.gt_ys = list()
        for i in range(5):
            image = tf.image.resize_images(self.x_flow_images, (int(self.dims.input_image[0]/(math.pow(2,6-i))),
                                                                int(self.dims.input_image[1]/(math.pow(2,6-i)))))
            self.gt_ys.append(tf.transpose(image, [0,3,1,2]))


    # Architecture #
    #--------------#

    def feature_pyramid_forward_pass(self, I):

        ct0_5 = tf.layers.conv2d(I,16,[3,3],(2,2),'same','channels_first',(1,1),tf.nn.leaky_relu)
        ct1 = tf.layers.conv2d(ct0_5, 16, [3, 3], (1, 1), 'same', 'channels_first', (1, 1), tf.nn.leaky_relu)

        ct1_5 = tf.layers.conv2d(ct1,32,[3,3],(2,2),'same','channels_first',(1,1),tf.nn.leaky_relu)
        ct2 = tf.layers.conv2d(ct1_5, 32, [3, 3], (1, 1), 'same', 'channels_first', (1, 1), tf.nn.leaky_relu)

        ct2_5 = tf.layers.conv2d(ct2,64,[3,3],(2,2),'same','channels_first',(1,1),tf.nn.leaky_relu)
        ct3 = tf.layers.conv2d(ct2_5, 64, [3, 3], (1, 1), 'same', 'channels_first', (1, 1), tf.nn.leaky_relu)

        ct3_5 = tf.layers.conv2d(ct3,64,[3,3],(2,2),'same','channels_first',(1,1),tf.nn.leaky_relu)
        ct4 = tf.layers.conv2d(ct3_5, 64, [3, 3], (1, 1), 'same', 'channels_first', (1, 1), tf.nn.leaky_relu)

        ct4_5 = tf.layers.conv2d(ct4,64,[3,3],(2,2),'same','channels_first',(1,1),tf.nn.leaky_relu)
        ct5 = tf.layers.conv2d(ct4_5, 64, [3, 3], (1, 1), 'same', 'channels_first', (1, 1), tf.nn.leaky_relu)

        ct5_5 = tf.layers.conv2d(ct5,64,[3,3],(2,2),'same','channels_first',(1,1),tf.nn.leaky_relu)
        ct6 = tf.layers.conv2d(ct5_5, 64, [3, 3], (1, 1), 'same', 'channels_first', (1, 1), tf.nn.leaky_relu)

        return [ct6, ct5, ct4, ct3, ct2, ct1]

    def flow_estimator_forward_pass(self, ctm1, ct, w):

        ct_trans = tf.transpose(ct, [0,2,3,1])
        w_trans = tf.transpose(w, [0,2,3,1])

        cwt_trans = self.warp(ct_trans,w_trans)

        cwt = tf.transpose(cwt_trans, [0,3,1,2])

        cvt = self.cost_volume(cwt,ctm1)

        concatted = tf.concat((cvt,w,ctm1),-3)

        conv1 = tf.layers.conv2d(concatted,128,[3,3],(1,1),'same','channels_first',(1,1),tf.nn.leaky_relu)
        conv2 = tf.layers.conv2d(conv1,128,[3,3],(1,1),'same','channels_first',(1,1),tf.nn.leaky_relu)
        conv3 = tf.layers.conv2d(conv2,96,[3,3],(1,1),'same','channels_first',(1,1),tf.nn.leaky_relu)
        conv4 = tf.layers.conv2d(conv3,64,[3,3],(1,1),'same','channels_first',(1,1),tf.nn.leaky_relu)
        ft = tf.layers.conv2d(conv4,32,[3,3],(1,1),'same','channels_first',(1,1),tf.nn.leaky_relu)
        wt = tf.layers.conv2d(ft,2,[3,3],(1,1),'same','channels_first',(1,1),tf.nn.leaky_relu)

        return ft, wt

    def context_forward_pass(self, ft, wt):

        concatted = tf.concat((ft, wt), -3)

        conv1 = tf.layers.conv2d(concatted,128,[3,3],(1,1),'same','channels_first',(1,1),tf.nn.leaky_relu)
        conv2 = tf.layers.conv2d(conv1,128,[3,3],(1,1),'same','channels_first',(2,2),tf.nn.leaky_relu)
        conv3 = tf.layers.conv2d(conv2,128,[3,3],(1,1),'same','channels_first',(4,4),tf.nn.leaky_relu)
        conv4 = tf.layers.conv2d(conv3,96,[3,3],(1,1),'same','channels_first',(8,8),tf.nn.leaky_relu)
        conv5 = tf.layers.conv2d(conv4,64,[3,3],(1,1),'same','channels_first',(16,16),tf.nn.leaky_relu)
        conv6 = tf.layers.conv2d(conv5,32,[3,3],(1,1),'same','channels_first',(1,1),tf.nn.leaky_relu)
        conv7 = tf.layers.conv2d(conv6,2,[3,3],(1,1),'same','channels_first',(1,1),tf.nn.leaky_relu)

        return conv7 + wt

    def define_network_structure(self):

        I1 = self.x_images[:,0]
        I2 = self.x_images[:,1]

        self.features1 = self.feature_pyramid_forward_pass(I1)
        self.features2 = self.feature_pyramid_forward_pass(I2)

        self.flow_terms = list()
        self.upsampled_flow_terms = list()
        self.flow_features = list()

        for i, features in enumerate(zip(self.features1,self.features2)):

            feature1 = features[0]
            feature2 = features[1]

            if i == 0:
                wt_upsampled = tf.zeros((self.batch_size,2,6,7))
                wt_upsampled_scaled = wt_upsampled
                self.flow_terms.append(wt_upsampled)
            else:
                wt = self.flow_terms[i]
                wt_trans = tf.transpose(wt, [0, 2, 3, 1])
                wt_trans_upsampled = tf.image.resize_images(wt_trans, ((int(wt.shape[-2] * 2), int(wt.shape[-1] * 2))))
                wt_upsampled = tf.transpose(wt_trans_upsampled, [0, 3, 1, 2])
                wt_upsampled_scaled = wt_upsampled*20/math.pow(2,6-i)

            self.upsampled_flow_terms.append(wt_upsampled)
            flow_features, estimated_flow = self.flow_estimator_forward_pass(feature1,feature2,wt_upsampled_scaled)
            self.flow_features.append(flow_features)

            refined_flow = self.context_forward_pass(flow_features, estimated_flow)

            self.flow_terms.append(wt_upsampled + estimated_flow + refined_flow)

        self.network_outputs = self.flow_terms[1:]


    # Build #
    #-------#

    def build_model(self):

        self.define_data_loader()

        self.define_network_inputs()
        self.define_network_targets()

        self.define_network_structure()
        self.define_cost()
        self.define_optimiser()
        self.define_optimisation()

        tf.global_variables_initializer().run()

    def load_params(self, sess, saver, chkpt_num=None, test=False):
        dir(tf.contrib)
        if chkpt_num is None:
            try:
                saver.restore(sess, tf.train.latest_checkpoint(self.chkpt_dir))
            except:
                return False, 0
            with open(self.chkpt_dir + 'checkpoint', 'r') as checkpoint_file:
                starting_it = int(checkpoint_file.read().split('\n', 1)[0].split('-', 1)[-1][:-1]) + 1
        else:
            try:
                saver.restore(sess, self.chkpt_dir + 'model-' + str(chkpt_num))
            except:
                return False, 0
            starting_it = chkpt_num + 1
        return True, starting_it


    # Optimisation #
    #--------------#

    def define_cost(self):

        self.n_targets = self.network_outputs

        # cost
        alphas = [0.32,0.08,0.02,0.01,0.005]
        self.cost_target = tf.constant([0.])
        for i in range(5):
            cost_target_terms = tf.pow(self.n_targets[i]-self.gt_ys[i],2)
            self.cost_target += alphas[i]*tf.reduce_sum(cost_target_terms)

        self.regularisation_loss = tf.constant([0.])
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for weight in weights:
            self.regularisation_loss += tf.nn.l2_loss(weight)
        factor_l2 = 0.0004

        self.cost_target += factor_l2*self.regularisation_loss
        self.cost_aux = tf.identity(float(0))
        self.cost = tf.squeeze(tf.add(self.cost_target, self.cost_aux))

        self.cost_last_step = self.cost

    def define_optimiser(self):

        self.learning_rate = tf.minimum(tf.maximum(
            tf.train.exponential_decay(self.initial_learning_rate, self.global_step,
            self.learning_decrement_rate, self.learning_decrement, staircase=True),
            self.min_learning_rate),self.max_learning_rate)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

    def define_optimisation(self):

        self.updating_train_step = self.optimizer.minimize(self.cost)
        self.train_step = tf.cond(self.vis_mode, lambda: True, lambda: self.updating_train_step)

        self.finished = tf.constant(False,tf.bool)


    # Log Summaries #
    #---------------#

    def init_summary(self):

        self.cost_summary = tf.summary.scalar('cost', self.cost)
        self.learning_rate_summary = tf.summary.scalar('learning_rate', self.learning_rate)
        self.log_summary_op = tf.summary.merge([self.cost_summary, self.learning_rate_summary])

        self.random_batch_num = tf.random_uniform([], 0, self.batch_size, tf.int32)
        self.vis_summary_op = tf.Summary()

        self.gpu_usage = tf.placeholder(tf.int32)
        self.cpu_usage = tf.placeholder(tf.int32)
        self.ram_usage = tf.placeholder(tf.float32)
        self.throughput = tf.placeholder(tf.float32)

        gpu_memory_summary = tf.summary.scalar('percent gpu memory',
                                tf.divide(tf.cast(tf.contrib.memory_stats.BytesInUse(),tf.float32),tf.constant(8e7)))
        gpu_usage_summary = tf.summary.scalar('percent gpu usage', self.gpu_usage)
        cpu_usage_summary = tf.summary.scalar('percent cpu usage', self.cpu_usage)
        ram_usage_summary = tf.summary.scalar('percent ram usage', self.ram_usage)
        throughput_summary = tf.summary.scalar('throughput MB', self.throughput)

        self.perflog_summary_op = tf.summary.merge([gpu_memory_summary,
                                                           gpu_usage_summary,
                                                           cpu_usage_summary,
                                                           ram_usage_summary,
                                                           throughput_summary])

    def get_image_summary(self, sess, dict_feed, fps):

        dict_feed[self.step_placeholder] = 0

        x_images_out, gt_flow_out, predicted_flow_out = \
            sess.run((self.x_images[self.random_batch_num], self.x_flow_images[self.random_batch_num],
                      self.network_outputs[-1][self.random_batch_num]), dict_feed)

        x_images_trans = np.transpose(x_images_out, (0,2,3,1))
        predicted_flow_trans = np.transpose(predicted_flow_out, (1,2,0))

        images_arr = native_ops.modify_images_for_vis(x_images_trans, gt_flow_out, predicted_flow_trans)

        return native_ops.convert_array_to_gif_summary(images_arr, tag='images', fps=fps)

    def get_log_summary(self, sess, dict_feed):
        return sess.run(self.log_summary_op, dict_feed)

    def get_summary(self, sess, summary_op, dict_feed):
        if summary_op is self.vis_summary_op:
            fps = 2
            return self.get_image_summary(sess, dict_feed, fps)
        elif summary_op is self.log_summary_op:
            return self.get_log_summary(sess, dict_feed)

    def write_summary(self, step, summary, writer):
        writer.add_summary(summary, step)
        writer.flush()

    def write_summaries(self, sess, i, dict_feed, summary_op, data_handles, summary_writers):

        training_handle = data_handles[0]
        validation_handle = data_handles[1]

        dict_feed[self.data_type] = training_handle

        training_summ = self.get_summary(sess, summary_op, dict_feed)
        self.write_summary(i,training_summ,summary_writers[0])

        dict_feed[self.data_type] = validation_handle

        validation_summ = self.get_summary(sess, summary_op, dict_feed)
        self.write_summary(i,validation_summ,summary_writers[1])
