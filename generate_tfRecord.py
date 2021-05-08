from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow.compat.v1 as tf

from PIL import Image
# from object_detection.utils import dataset_util
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# 将分类名称转成ID号
def class_text_to_int(row_label):
    if row_label == '0':
        return 1
    elif row_label == '1':
        return 2
    elif row_label == '2':
        return 3
    elif row_label == '3':
        return 4
    elif row_label == '8':
        return 5
    elif row_label == '10':
        return 6
    elif row_label == '20':
        return 7
    elif row_label == '31':
        return 8
    elif row_label == '32':
        return 9
    elif row_label == '33':
        return 10
    elif row_label == '34':
        return 11
    elif row_label == '42':
        return 12
    elif row_label == '60':
        return 13
    else:
        print('NONE: ' + str(row_label))
        # None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    print(os.path.join(path, '{}'.format(group.filename)))
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = (group.filename + '.png').encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        # xmins.append(float(row['xmin'] / width))
        # xmaxs.append(float(row['xmax'] / width))
        # ymins.append(float(row['ymin'] / height))
        # ymaxs.append(float(row['ymax'] / height))
        xmn = float(row['xmin'] / width)
        if xmn <= 0.0:
            xmn = 0.0
        elif xmn >= 1.0:
            xmn = 1.0
        xmins.append(xmn)

        xmx = float(row['xmax'] / width)
        if xmx <= 0.0:
            xmx = 0.0
        elif xmx >= 1.0:
            xmx = 1.0
        xmaxs.append(xmx)

        ymn = float(row['ymin'] / height)
        if ymn <= 0.0:
            ymn = 0.0
        elif ymn >= 1.0:
            ymn = 1.0
        ymins.append(ymn)

        ymx = float(row['ymax'] / height)
        if ymx <= 0.0:
            ymx = 0.0
        elif ymx >= 1.0:
            ymx = 1.0
        ymaxs.append(ymx)
        print(xmn, xmx, ymn, ymx)
        clas = str(row['class'])
        classes_text.append(clas.encode('utf8'))
        classes.append(class_text_to_int(clas))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(csv_input, output_path, imgPath):
    writer = tf.python_io.TFRecordWriter(output_path)
    path = imgPath
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    # imgPath = 'E:\data\Images'
    imgPath = '/home/mxt/di/dataset/project_data/algorithm_data/ats_sz/data_processed/dataset_series/poly/ng_rough/20210322'

    # 生成train.record文件
    output_path = '/home/mxt/mxt/data/train.record'
    csv_input = '/home/mxt/mxt/data/train.csv'
    main(csv_input, output_path, imgPath)

    # 生成验证文件 eval.record
    # output_path = 'data/eval.record'
    # csv_input = 'data/eval.csv'
    # main(csv_input, output_path, imgPath)


