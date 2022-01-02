# -*- coding: utf-8 -*-
"""Data Loader"""

import os, sys
sys.path.append('.')

# #external
# import jsonschema
import cv2

# internal 
from configs.module.config import CFG
# from configs.module.data_schema import SCHEMA


class DataLoader:
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