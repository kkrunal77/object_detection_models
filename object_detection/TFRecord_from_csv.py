import hashlib
import io
import logging
import os
import random
import re
import pandas as pd

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/kk/Downloads/infilect/my_data/', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '/home/kk/Downloads/infilect/my_data/tf_record/', 'Path to directory to output TFRecords.')
FLAGS = flags.FLAGS

def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False):
  #data.image_name.apply(str)
  #print(str(data['image_name']))
  #print(str(data['image_name'].values[0]))
  img_path = os.path.join(image_subdirectory, str(data['image_name'].values[0]))
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)

  if image.format not in ('JPEG', 'PNG'):
    print(image.format)
    print("*************",img_path)
    raise ValueError('Image format not JPEG')

  key = hashlib.sha256(encoded_jpg).hexdigest()
  width = int(image.size[0])
  height = int(image.size[1])
  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  poses = []
  xmin.append(float(data['xmin'].values[0]) / width)
  ymin.append(float(data['ymin'].values[0]) / height)
  xmax.append(float(data['xmax'].values[0]) / width)
  ymax.append(float(data['ymax'].values[0]) / height)
  class_name = data['classes'].values[0]
  classes_text.append(label_map_dict[class_name].encode('utf8'))
  classes.append(class_name)
  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['image_name'].values[0].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['image_name'].values[0].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
  return example

def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):
  writer = tf.python_io.TFRecordWriter(output_filename)
  path = os.path.join(annotations_dir, 'annotations.csv')
  annotation_data=pd.read_csv(path)
  for idx, example in enumerate(examples):
    if idx % 100 == 0:
      print('On image {} of {}'.format(idx, len(examples)))
      print(example)
    if not os.path.exists(path):
      print('Could not find %s, ignoring example.', path)
      continue
    # try:
    data=annotation_data.loc[(annotation_data['image_name'] == example)]
    if len(data.values) == 0:
      print(example)
    tf_example = dict_to_tf_example(data, label_map_dict, image_dir)
    writer.write(tf_example.SerializeToString())
  writer.close()


def main(_):
  data_dir = FLAGS.data_dir
  label_map_dict= {1: 'box'}

  image_dir = os.path.join(data_dir, 'images/')
  annotations_dir = os.path.join(data_dir, 'annotations/')
  traintest_path = os.path.join(annotations_dir, 'annotations.csv')

  traintest=pd.read_csv(traintest_path)
  
  train_examples = traintest.loc[(traintest.evaluation_status != 'val'), ['image_name']]['image_name'].tolist()
  # print(train_examples)
  val_examples = traintest.loc[(traintest.evaluation_status == 'val'), ['image_name']]['image_name'].tolist()
  train_output_path = os.path.join(FLAGS.output_dir, 'table_train.record')
  # print(train_output_path)
  val_output_path = os.path.join(FLAGS.output_dir, 'table_val.record')
  # print(len(val_examples))
  create_tf_record(train_output_path, label_map_dict, annotations_dir,image_dir, train_examples)
  create_tf_record(val_output_path, label_map_dict, annotations_dir,image_dir, val_examples)
  
if __name__ == '__main__':
    tf.app.run()


