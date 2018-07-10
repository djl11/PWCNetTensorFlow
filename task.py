import os

from trainer import Trainer
from data_loader import DataLoader
from directories import Directories
from network import Network

class Task():

    def __init__(self):

        self._dirs = Directories()
        self._init_data_loader()
        self._init_network()

    def _update_dirs(self, log_folder):

        current_dir = os.path.dirname(os.path.realpath(__file__))
        network_dir = Network.directory() + log_folder
        self._dirs.update_for_task(current_dir, network_dir)

    def _init_network(self):

        self.__network = Network(data_loader=self.__data_loader,
                                 dirs=self._dirs)

    def _init_data_loader(self):
        self.__data_loader = DataLoader(dirs=self._dirs,
                                        total_num_examples=22872)

    def _init_trainer(self):

        self.__trainer = Trainer(data_loader=self.__data_loader,
                                 network=self.__network,
                                 dirs=self._dirs,
                                 ld_chkpt=True,
                                 save_freq=100,
                                 log_freq=20,
                                 vis_freq=50)
        return self.__trainer


    def run(self, train):

        self._init_trainer()
        log_folder = '/logged_data'

        self._update_dirs(log_folder)
        self.__network.chkpt_dir = self._dirs.chkpt_dir

        if train:
            self.__trainer.run_trainer()
        else:
            self.__trainer.run_visualiser()
        self.__trainer.sess.close()