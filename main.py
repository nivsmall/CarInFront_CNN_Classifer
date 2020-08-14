import FilterData
import Labelling
import data_processing
import input_pipeline
import train
print('hello world')


#data = FilterData.filterData(['clear', 'overcast', 'undefined', 'partly', 'cloudy'], ['tunnel', 'highway'],
#                             ['daytime', 'dawn/dusk', 'undefined'], 'bdd100k/labels/bdd100k_labels_images_test.json')
#FilterData.preview_filtered_data(data, 'bdd100k/bdd100k/images/100k/test')
#FilterData.relocate_wanted_images(data,'bdd100k/bdd100k/images/100k/test', 'bdd100k/used_images/test')

#Labelling.ManualLabelling(DATA_DIR="bdd100k/used_images/train/")
#Labelling.labels_Json_from_image_names("bdd100k/used_images/val/", 'bdd100k/used_labels/val_labels_1.json')

#labels_dict = data_processing.load_labels('bdd100k/used_labels/val_labels.json')
#examples_cnt = data_processing.write_tfrecord('bdd100k/tfrecords/val_test.tfrecords', labels_dict, 'bdd100k/used_images/val/')
#print(examples_cnt)

#input_pipeline.read_tfrecord('bdd100k/tfrecords/train_test.tfrecords', show=True, shuffle=True)

