import numpy as np
import json

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class WheatDataset(CustomDataset):

    CLASSES = ('wheat',)

    def load_annotations(self, ann_file):
        with open(ann_file, 'r') as fp:
            config = json.load(fp)

        data_infos = []
        for img in config['images']:
            bboxes =[]
            labels = []
            for annotation in config['annotations']:
                if img['id'] == annotation['image_id']:
                    xmin, ymin, w, h = annotation['bbox']
                    bboxes.append([xmin, ymin, xmin+w, ymin+h])
                    labels.append(annotation['category_id'])
                
            data_infos.append(
                dict(
                    filename=img['file_name'],
                    width=img['width'],
                    height=img['height'],
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64))
            ))
            

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']