import os, json, random
from tqdm import tqdm
from loguru import logger
import cv2


class CocoData:
    def __init__(self):
        self.categories = []
        self.images = []

    def from_file(self, coco_data_file):
        coco_json_data = json.load(open(coco_data_file))
        self.from_json_data(coco_json_data)

    def from_json_data(self, coco_json_data):
        """
        coco_data={
            'categories': [{'id': cat_id, 'name': cat_label, 'supercategory': 'none'}],
            'images': [{'id': img_id, 'width': img_width, 'height': img_height, 'file_name': file_name, 
                         'anns': [{'id': ann_id, 'image_id': img_id, 'category': cat_id, 'area': bbox_area, 'bbox': [x0, y0, w, h]}] }]
        }
        """
        categories = coco_json_data['categories']
        images = coco_json_data['images']
        for img in images:
            img['anns'] = []
        annotations = coco_json_data['annotations']

        map_images = {img['id']: img for img in images}
        for ann in tqdm(annotations):
            img_id = ann['image_id']
            assert img_id in map_images
            img = map_images[img_id]
            img['anns'].append(ann)

        self.categories = categories
        self.images = images

    def save(self, file_name):
        coco_images = []
        coco_annotations = []
        for img in tqdm(self.images):
            coco_images.append({
                'id': img['id'],
                'width': img['width'],
                'height': img['height'],
                'file_name': img['file_name']
            })
            coco_annotations.extend(img['anns'])
        coco_json_data = {
            'categories': self.categories,
            'images': coco_images,
            'annotations': coco_annotations
        }

        json.dump(coco_json_data,
                  open(file_name, 'w'),
                  ensure_ascii=False,
                  indent=2)
        logger.info(f"Saved {len(self.images)} images in {file_name}")

    def split(self, train_ratio=0.9):
        num_images = len(self.images)
        num_train_images = int(num_images * train_ratio)
        image_indices = list(range(num_images))
        train_indices = random.sample(image_indices, num_train_images)
        train_images = []
        val_images = []
        for idx, img in enumerate(tqdm(self.images)):
            if idx in train_indices:
                train_images.append(img)
            else:
                val_images.append(img)
        train_coco_data = CocoData()
        train_coco_data.categories = self.categories
        train_coco_data.images = train_images
        val_coco_data = CocoData()
        val_coco_data.categories = self.categories
        val_coco_data.images = val_images

        return train_coco_data, val_coco_data
