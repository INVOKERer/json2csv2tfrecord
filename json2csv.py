# xml2csv.py

import os
import glob
import pandas as pd
import json
import tqdm
os.chdir('/home/mxt/di/dataset/project_data/algorithm_data/ats_sz/data_processed/dataset_series/poly/ng_rough/20210322')
path = '/home/mxt/di/dataset/project_data/algorithm_data/ats_sz/data_processed/dataset_series/poly/ng_rough/20210322'


def json_to_csv(path):
    json_list = []
    for json_file in tqdm.tqdm(glob.glob(path + '/*.json')):
        with open(json_file, 'r') as load_f:
            load_dict = json.load(load_f)
        # print(load_dict)
        for member in load_dict['shapes']:
            x1, x2 = float(member['points'][0][0]), float(member['points'][1][0])
            y1, y2 = float(member['points'][0][1]), float(member['points'][1][1])
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            value = (path + '/' + load_dict['imagePath'],
                     float(load_dict['imageHeight']),
                     float(load_dict['imageWidth']),
                     str(member['label']),
                     xmin,
                     ymin,
                     xmax,
                     ymax
                     )
            json_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(json_list, columns=column_name)
    return xml_df


def main():
    image_path = path
    json_df = json_to_csv(image_path)
    # json_df.to_csv('train.csv', index=None, encoding='utf-8')
    # xls_die = '/home/mxt/di/dataset/project_data/algorithm_data/ats_sz/data_processed/doc_series/poly/20210416/dataset.xls'
    # data = pd.read_excel(xls_die, 'data')
    # data.to_csv('data.csv', encoding='utf-8')
    json_df.to_csv('/home/mxt/mxt/data/train.csv', index=None, encoding='utf-8')
    print('Successfully converted json to csv.')


main()