import os
import cv2
import json
from FilterData import relocate_wanted_images
from data_processing import load_labels


def ManualLabelling(DATA_DIR):
    ''''
            Label class of image: (manually)
                No car at lane at all           _#0
                Car in front, close distance    _#1
                Car at moderate distance        _#2
                Car at distance                 _#3
                Not Sure/Undefined              _#5
    '''

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


def labels_Json_from_image_names(img_dir, Json_dir):
    labeled_dict = {}
    with os.scandir(img_dir) as scan:
        for img in scan:
            if '_#' in img.name:
                pre_name = img.name.split('_#')[0] + '.jpg'
                label = img.name.split('_#')[1][0]
                labeled_dict[pre_name] = label
                if label == '0':
                    os.rename(os.path.join(img_dir, img.name), os.path.join(img_dir, '0/', pre_name))
                if label == '1':
                    os.rename(os.path.join(img_dir, img.name), os.path.join(img_dir, '1/', pre_name))

    with open(Json_dir, 'w') as json_file:
        json.dump(labeled_dict, json_file)
    return
