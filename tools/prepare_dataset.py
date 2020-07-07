import numpy as np
import pandas as pd
import ast
import json
import argparse
from sklearn.model_selection import train_test_split
import time

def get_image_list(df):

    images=[]
    for _, row in df.iterrows():
        images.append({
           'file_name':row.image_id+'.jpg',
            'height': row.height,
            'width': row.width,
            'id': row.id
        })
    return images


def get_annotations_list(df):
    annotations = []
    for i, (_, row) in enumerate(df.iterrows()):
        annotations.append({
            'image_id': row.id,
            'bbox': ast.literal_eval(row.bbox),
            'category_id': row.category_id,
            'id':i
        })
    return annotations

def create_train_test_dataset(filename, test_size=0.2, target_directory='data/'):
    images_df = pd.read_csv(filename)
    print("Number of rows in file: {}".format(len(images_df)))

    #add unique id for each image
    image_ids = images_df['image_id'].unique().tolist()
    ids = list(range(len(image_ids)))
    image_ids_dict = dict(zip(image_ids, ids))
    images_df['id'] = images_df['image_id'].map(image_ids_dict)
    images_df['category_id'] = 0

    #create dataframe of unique image ids
    unique_images_df = images_df[['image_id', 'width', 'height', 'id']].drop_duplicates()
    print("Number of images with bounding boxes: {}".format(len(unique_images_df)))

    #split images into train and test split
    train_df, val_df = train_test_split(unique_images_df, test_size=test_size, random_state=40)

    print("Number of images in train set: {}".format(len(train_df)))
    print("Number of images in val set: {}".format(len(val_df)))

    train_images = get_image_list(train_df)
    val_images = get_image_list(val_df)

    train_annotations = get_annotations_list(images_df[images_df['image_id'].isin(train_df['image_id'].tolist())])
    val_annotations = get_annotations_list(images_df[images_df['image_id'].isin(val_df['image_id'].tolist())])

    categories= [ {'id': 0, 'name': 'wheat'}]
    config_train = {'images': train_images, 'annotations': train_annotations, 'categories': categories}
    config_val = {'images': val_images, 'annotations': val_annotations, 'categories': categories}

    
    with open(target_directory+'wheat_config_train.json', 'w') as fp:
        json.dump(config_train, fp)
        
    with open(target_directory+'wheat_config_val.json', 'w') as fp:
        json.dump(config_val, fp)

    print("train and val config files created in directory: {}".format(target_directory))    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", required=True, help="file name of csv file containing annotations")
    parser.add_argument("-t", "--test_size", required=False, help='size of test dataset as fraction of the total size', type=float)
    parser.add_argument("-td", "--target_directory", required=False, help="directory to save train and val config")

    args = vars(parser.parse_args())
    assert args['filename'], "Please provide the name of annotations file"

    filename = args['filename']
    
    test_size = 0.2
    target_directory = 'data/'

    if args['test_size']:
        test_size= args['test_size']

    if args['target_directory']:
        target_directory = args['target_directory']    
    start = time.time()
    create_train_test_dataset(filename, test_size, target_directory)
    print("Total time taken: {:.2f} seconds".format(time.time()-start))   