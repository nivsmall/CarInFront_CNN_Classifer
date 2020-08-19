import os
import cv2
import json
import numpy as np
from FilterData import relocate_wanted_images
from data_processing import load_labels


def ManualLabelling(DATA_DIR):
    """
    A simplistic user interface to manually label images
    Class of labeled image will be added to the file name:
        Label class of image: (manually)
                    No car at lane at all           _#0
                    Car in front, close distance    _#1
                    Car at moderate distance        _#2 won't be used
                    Car at distance                 _#3 won't be used
                    Not Sure/Undefined              _#5 won't be used
    :param DATA_DIR: Images directory
    :return: None
    """
    with os.scandir(DATA_DIR) as scan:
        for img in scan:
            print(img.name)
            if '_#' in img.name:
                continue
            mat = cv2.imread(os.path.join(DATA_DIR, img.name))
            cv2.imshow('Assign Label to Image: (9 to exit)', mat)
            label = chr(cv2.waitKey())
            cv2.destroyAllWindows()
            if label == '9': break
            labeled_name = img.name.split('.jpg')[0] + '_#' + label + '.jpg'
            os.rename(os.path.join(DATA_DIR, img.name), os.path.join(DATA_DIR, labeled_name))
        return


def labels_Json_from_image_names(img_dir, Json_path):
    """
    After labels were assigned and file name modified accordingly (using ManualLabelling):
        Move labeled images to class folder and create a Json file containing (image-name & class) pairs
    :param img_dir:
    :param Json_path: Json path to be written
    :return: None
    """
    labeled_dict = {}
    img_dir0 = os.path.join(img_dir, '0/')
    img_dir1 = os.path.join(img_dir, '1/')
    if not os.path.isdir(img_dir0): os.mkdir(img_dir0)
    if not os.path.isdir(img_dir1): os.mkdir(img_dir1)

    with os.scandir(img_dir) as scan:
        for img in scan:
            if '_#' in img.name:
                pre_name = img.name.split('_#')[0] + '.jpg'
                label = img.name.split('_#')[1][0]
                labeled_dict[pre_name] = label
                if label == '0':
                    os.rename(img_dir0, os.path.join(img_dir0, pre_name))
                if label == '1':
                    os.rename(img_dir1, os.path.join(img_dir1, pre_name))

    with open(Json_path, 'w') as json_file:
        json.dump(labeled_dict, json_file)
    return

