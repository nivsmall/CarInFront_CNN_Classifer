import FilterData
import Labelling
import data_processing
import input_pipeline
from train import train
print('hello world')


# Filter images to keep wanted attributes only:
data = FilterData.filterData(['clear', 'overcast', 'undefined', 'partly', 'cloudy'], ['tunnel', 'highway', 'residential', 'undefined', 'city street'],
                             ['daytime', 'dawn/dusk'], 'bdd100k/labels/bdd100k_labels_images_val.json')
FilterData.preview_filtered_data(data, 'bdd100k/used_images/val/')

# Relocate filtered images:
FilterData.relocate_wanted_images(data, 'bdd100k/bdd100k/images/100k/val/', 'bdd100k/used_images/val/')

# Manually label data: (if needed)
Labelling.ManualLabelling(DATA_DIR="bdd100k/used_images/val/")

# Separate data by class into folders:
Labelling.move_images_to_class_folders("bdd100k/used_images/val/")

# Create Json file to hold labels from split folders
Labelling.labels_Json_from_image_paths("bdd100k/used_images/val/", 'bdd100k/used_labels/val.json')

# Write a tf.record file using image-label pairs from Json file
labels_dict = data_processing.load_labels('bdd100k/used_labels/val.json')
examples_cnt = data_processing.write_tfrecord('bdd100k/tfrecords/val.tfrecords', labels_dict,
                                              'bdd100k/used_images/val/', False)
print(examples_cnt)

# Verify data before training:
input_pipeline.read_tfrecord('bdd100k/tfrecords/train.tfrecords', show=True)

# augment data:
data_processing.balance_data_via_augmentation('bdd100k/used_images/train/', False)

train('bdd100k/tfrecords/train.tfrecords', transfer=False)


