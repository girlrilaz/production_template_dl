# -*- coding: utf-8 -*-
"""Data Loader unit testing"""

import os, sys
sys.path.append('.')

# #external
# import jsonschema
from unittest.mock import patch
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

# internal 
from configs.module.config import CFG
# from configs.module.data_schema import SCHEMA

def dummy_load_data(*args, **kwargs):
    with tfds.testing.mock_data(num_examples=1):
        return tfds.load(CFG['data']['path'], with_info=True)

class DataLoaderTest:
    """Data Loader class"""

    @staticmethod
    def load_data(data_config, filenames):
        """
        Loads dataset from path and process with cv2
        Args:
        data_config: configs from the config file  
        """

        # Images are stored under the images/ directory
        data = []
        for filename in filenames:
            img_filepath = f'{os.path.join(data_config, filename)}'

            # Read the image into a numpy array and append in images list
            data.append(cv2.imread(img_filepath))

        return data

    @patch('models.unet.DataLoader.load_data')
    def test_load_data(self, mock_data_loader):

        mock_data_loader.side_effect = dummy_load_data
        shape = tf.TensorShape([None, self.unet.image_size, self.unet.image_size, 3])

        self.unet.load_data()
        mock_data_loader.assert_called()

        self.assertItemsEqual(self.unet.train_dataset.element_spec[0].shape, shape)
        self.assertItemsEqual(self.unet.test_dataset.element_spec[0].shape, shape)
    
if __name__ == '__main__':
    
    data = DataLoader()

    image_files = [
                'apple.jpg',
                'clock.jpg',
                'oranges.jpg',
                'car.jpg'
            ]

    data = data.load_data(CFG['data']['path'], image_files)

    for image_file in data:
        print(image_file)



