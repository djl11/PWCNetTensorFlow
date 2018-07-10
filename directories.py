import os

class Directories:

    def __init__(self):
        self.init_directories()

    def init_directories(self):
        self.__project_dir = os.getcwd()
        self.__chkpt_dir = '/chkpts/'
        self.__log_dir = '/log/'
        self.__rgb_image_dir = '/FlyingChairs/data/'
        self.__flow_image_dir = '/FlyingChairs/data/'
        self.__rgb_format = '.ppm'
        self.__flow_format = '.flo'
        self.__training_dir = 'training/'
        self.__validation_dir = 'validation/'
        self.__networks_dir = '/networks/'

    def update_for_task(self, current_dir, network_dir):

        self.init_directories()

        self.__networks_dir = current_dir + self.__networks_dir
        self.__chkpt_dir = network_dir + self.__chkpt_dir
        self.__log_dir = network_dir + self.__log_dir
        self.__rgb_image_dir = current_dir + self.__rgb_image_dir
        self.__flow_image_dir = current_dir + self.__flow_image_dir

    # Getters #
    #---------#

    @property
    def project_dir(self):
        return self.__project_dir
    @property
    def networks_dir(self):
        return self.__networks_dir
    @property
    def chkpt_dir(self):
        return self.__chkpt_dir
    @property
    def log_dir(self):
        return self.__log_dir
    @property
    def rgb_image_dir(self):
        return self.__rgb_image_dir
    @property
    def flow_image_dir(self):
        return self.__flow_image_dir
    @property
    def image_data_filename(self):
        return self.__image_data_filename
    @property
    def rgb_format(self):
        return self.__rgb_format
    @property
    def flow_format(self):
        return self.__flow_format
    @property
    def training_dir(self):
        return self.__training_dir
    @property
    def validation_dir(self):
        return self.__validation_dir
