import numpy as np
import os
import tensorflow as tf
import custom_ops.compiled as compiled_ops
import urllib
import zipfile

class DataLoader:

    def __init__(self, dirs, total_num_examples):

        self.dirs = dirs
        self.total_num_examples = total_num_examples

        self.download_and_extract_data_if_necessary()

    def download_and_extract_data_if_necessary(self):
        cwd = os.getcwd()
        url = 'https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip'
        filepath = cwd + '/' + url.split('/')[-1]

        if not os.path.isdir(filepath[:-4]):

            result = input('About to download and extract the flying chairs dataset.'
                           '\nThis requires ~85GB of free space. Continue? [y/n]\n')

            if result == 'y':
                print('downloading flying chairs dataset as zip, this may take ~30-60 minutes...')
                urllib.request.urlretrieve(url, filepath)
            else:
                exit(0)

            print('extracting flying chairs dataset, this may take a while...')

            zip_ref = zipfile.ZipFile(filepath, 'r')
            zip_ref.extractall(cwd)
            zip_ref.close()
            os.remove(filepath)
            os.rename(filepath[:-4] + '_release', filepath[:-4])

    def input_parser(self, img_paths, flow_img_paths):

        imgs = tf.map_fn(lambda img_path: compiled_ops.decode_ppm(tf.read_file(img_path[0]))[0], img_paths, dtype=tf.uint8)
        flow_imgs = tf.map_fn(lambda flow_img_path: compiled_ops.decode_flo(tf.read_file(flow_img_path[0])), flow_img_paths, dtype=tf.float32)
        return imgs, flow_imgs

    def prime_image_data(self, image_dir, starting_example, ending_example, file_extension):

        image_filenames = [file for file in os.listdir(image_dir) if file.endswith(file_extension)]
        image_filenames.sort()
        image_paths = [image_dir + image_filename for image_filename in image_filenames]

        grouped_images = []
        group = []
        image_num = starting_example
        for image_path in image_paths:
            trimmed_string = image_path.split('/')[-1].split('_')[0]
            current_image_num = int(trimmed_string) - 1 if str.isdigit(trimmed_string) else int(trimmed_string[1:]) - 1
            if current_image_num >= starting_example:
                if current_image_num != image_num:
                    image_num = current_image_num
                    if current_image_num > ending_example:
                        break
                    grouped_images.append(group)
                    group = []
                group.append(image_path)

        grouped_images.append(group)

        return grouped_images


    def prime_data_for_loading(self, starting_example, ending_example, batch_size, training):

        self.batch_size = batch_size

        image_paths_list = list()
        image_paths_tensor_list = list()

        rgb_image_dir = self.dirs.rgb_image_dir
        grouped_rgb_images = self.prime_image_data(rgb_image_dir, starting_example, ending_example, self.dirs.rgb_format)
        rgb_img_paths_tensor = tf.expand_dims(tf.constant(grouped_rgb_images), -1)

        image_paths_list.append(grouped_rgb_images)
        image_paths_tensor_list.append(rgb_img_paths_tensor)

        flow_image_dir = self.dirs.flow_image_dir
        grouped_flow_images = self.prime_image_data(flow_image_dir, starting_example, ending_example, self.dirs.flow_format)
        flow_img_paths_tensor = tf.expand_dims(tf.constant(grouped_flow_images),-1)

        image_paths_list.append(grouped_flow_images)
        image_paths_tensor_list.append(flow_img_paths_tensor)

        data = tf.data.Dataset.from_tensor_slices(tuple(image_paths_tensor_list))
        data = data.map(map_func=self.input_parser, num_parallel_calls=8)
        data = data.shuffle(tf.cast(batch_size*100,tf.int64))
        data = data.apply(tf.contrib.data.batch_and_drop_remainder(tf.cast(batch_size,tf.int64)))
        data = data.prefetch(1)
        data = data.repeat()

        return data

    def prime_data(self, start_it_train, end_it_train, start_it_valid, end_it_valid, batch_size, data_type):

        training_dataset = self.prime_data_for_loading(start_it_train, end_it_train, batch_size, True)
        self.training_data_len = end_it_train - start_it_train

        validation_dataset = self.prime_data_for_loading(start_it_valid, end_it_valid, batch_size, False)
        self.validation_data_len = end_it_valid - start_it_valid

        iterator = tf.data.Iterator.from_string_handle(data_type, training_dataset.output_types, training_dataset.output_shapes)
        self.next_element = iterator.get_next()

        training_iterator = training_dataset.make_initializable_iterator()
        validation_iterator = validation_dataset.make_initializable_iterator()

        self.training_init_op = training_iterator.make_initializer(training_dataset)
        self.validation_init_op = validation_iterator.make_initializer(validation_dataset)

        self.training_handle = self.sess.run(training_iterator.string_handle())
        self.validation_handle = self.sess.run(validation_iterator.string_handle())

    def set_session(self, sess):
        self.sess = sess