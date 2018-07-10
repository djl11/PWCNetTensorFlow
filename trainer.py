import tensorflow as tf  # add tensorflow framework
import os
import libtmux
import pathlib

class Trainer():

    def __init__(self, data_loader, network, dirs, ld_chkpt, save_freq, log_freq,
                 vis_freq):

        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

        self.data_loader = data_loader
        self.data_loader.set_session(self.sess)

        self.network = network
        self.num_training_examples = self.network.num_training_examples
        self.num_validation_examples = self.network.num_validation_examples
        self.total_num_examples = self.num_training_examples + self.num_validation_examples

        self.networks_dir = dirs.networks_dir
        self.ld_chkpt = ld_chkpt

        self.save_freq = save_freq
        self.log_freq = log_freq
        self.vis_freq = vis_freq
        self.dirs = dirs

    def init_saver(self):
        self.chkpt_dir = self.dirs.chkpt_dir
        all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        variables_to_save = [variable for variable in all_variables if variable.name[0:11] != 'placeholder']
        self.saver = tf.train.Saver(var_list=variables_to_save, max_to_keep=None)
        pathlib.Path(self.chkpt_dir).mkdir(parents=True, exist_ok=True)

    def initial_save(self):
        if not os.path.exists(self.chkpt_dir + 'model.meta'):
            self.saver.save(self.sess, self.chkpt_dir + 'model')

    def make_log_dirs(self):
        for log_dir in self.log_dirs:
            pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    def init_log_dirs(self):

        self.log_dir = self.dirs.log_dir

        self.log_dir_train = self.log_dir + 'plots/' + self.dirs.training_dir
        self.log_dir_valid = self.log_dir + 'plots/' + self.dirs.validation_dir

        self.vis_dir_train = self.log_dir + 'vis/' + self.dirs.training_dir
        self.vis_dir_valid = self.log_dir + 'vis/' + self.dirs.validation_dir

        self.log_dirs = [
            self.log_dir_train,
            self.log_dir_valid,
            self.vis_dir_train,
            self.vis_dir_valid]

        self.make_log_dirs()

    def init_log_summary_writers(self):

        self.sw_log_train = tf.summary.FileWriter(self.log_dir_train, graph=tf.get_default_graph())
        self.sw_log_valid = tf.summary.FileWriter(self.log_dir_valid, graph=tf.get_default_graph())

        self.summary_writers[0] = self.sw_log_train
        self.summary_writers[1] = self.sw_log_valid

    def init_vis_summary_writers(self):

        self.sw_vis_train = tf.summary.FileWriter(self.vis_dir_train, graph=tf.get_default_graph())
        self.sw_vis_valid = tf.summary.FileWriter(self.vis_dir_valid, graph=tf.get_default_graph())

        self.summary_writers[2] = self.sw_vis_train
        self.summary_writers[3] = self.sw_vis_valid

    def init_summary_writers(self):

        self.network.init_summary()
        self.summary_writers = [0]*4

        self.init_log_summary_writers()
        self.init_vis_summary_writers()

    def init_logger(self):
        self.init_log_dirs()
        self.init_summary_writers()

    def init_dataset_loader(self, i):

        dict_feed = {self.network.data_type: self.data_loader.training_handle, self.network.global_step: i}
        self.sess.run(self.data_loader.training_init_op,dict_feed)

        dict_feed[self.network.data_type] = self.data_loader.validation_handle
        self.sess.run(self.data_loader.validation_init_op,dict_feed)

    def write_summaries(self, i, dict_feed, summary_op, summary_writers):
        data_handles = (self.data_loader.training_handle, self.data_loader.validation_handle)
        self.network.write_summaries(self.sess, i, dict_feed, summary_op, data_handles, summary_writers)

    def log(self, i):
        self.write_summaries(i, self.dict_feed, self.network.log_summary_op, self.summary_writers[0:2])
        print('logged, step ' + str(i))

    def vis(self, i):
        self.write_summaries(i, self.dict_feed, self.network.vis_summary_op, self.summary_writers[2:4])

    def save(self, i):
        self.saver.save(self.sess, self.chkpt_dir + 'model', global_step=i, write_meta_graph=False)
        print('saved, step ' + str(i))

    def train(self, starting_it, vis_mode=False):

        global_step = starting_it

        self.dict_feed = {}
        self.dict_feed[self.network.vis_mode] = vis_mode
        self.dict_feed[self.network.global_step] = global_step
        self.dict_feed[self.network.data_type] = self.data_loader.training_handle

        if starting_it == self.network.total_iterations: return True

        if vis_mode:
            self.vis_freq = 1
            self.dict_feed[self.network.vis_mode] = True

        while global_step < self.network.total_iterations or self.network.total_iterations == -1:

            self.dict_feed[self.network.global_step] = global_step
            self.dict_feed[self.network.data_type] = self.data_loader.training_handle
            self.dict_feed[self.network.step_placeholder] = 0

            _, finished, cost = self.sess.run((self.network.train_step, self.network.finished, self.network.cost_last_step), self.dict_feed)

            if global_step % self.log_freq == 0 and not vis_mode: self.log(global_step)
            if global_step % self.vis_freq == 0: self.vis(global_step)
            if global_step % self.save_freq == 0 and not vis_mode: self.save(global_step)

            global_step += 1

            if vis_mode: input('press enter to visualise another example')

        return True

    def start_tensorboard(self):
        self.stop_tensorboard()
        server = libtmux.Server()
        session_name = 'tensorboard'
        os.system('tmux new-session -s ' + session_name + ' -d')
        session = server.find_where({'session_name': session_name})
        self.tmux_window = session.attached_window
        pane = self.tmux_window.split_window(attach=False)
        port = 6006
        pane.send_keys('tensorboard --logdir=' + os.getcwd() + '/logged_data' + ' --port ' + str(port))

    def stop_tensorboard(self):
        try:
            server = libtmux.Server()
            session_name = 'tensorboard'
            session = server.find_where({'session_name': session_name})
            self.tmux_window = session.attached_window
            self.tmux_window.kill_window()
        except NameError:
            print('No session was running')
        except:
            return

    def prime_data(self):

        self.data_loader.prime_data(start_it_train=0,
                                    end_it_train=self.num_training_examples-1,
                                    start_it_valid=self.num_training_examples,
                                    end_it_valid=self.total_num_examples-1,
                                    batch_size=self.network.batch_size,
                                    data_type=self.network.data_type)

    def __init_data_and_model(self):

        self.prime_data()

        starting_iteration = 0
        self.network.build_model()
        self.init_saver()
        if self.ld_chkpt is True:
            load_success, starting_iteration = self.network.load_params(self.sess, self.saver)
            if load_success is False:
                print('model built')
            else:
                print('model loaded')
        else:
            print('model built')

        self.init_dataset_loader(starting_iteration)
        self.initial_save()
        self.start_tensorboard()

        return starting_iteration

    def run_trainer(self):
        print('started trainer')
        starting_iteration = self.__init_data_and_model()
        self.init_logger()
        return self.train(starting_iteration)

    def run_visualiser(self):
        print('started visualiser')
        self.__init_data_and_model()
        self.init_logger()
        self.train(0,True)
