import os.path as osp
import os

import mmcv
import numpy as np
import argparse
import imgaug.augmenters as iaa
import xml.etree.ElementTree as ET
import functools as ft
from PIL import Image

VOC_type = ""
VOCdevkit_dir = ""
CoCo_output_dir = ""

voc_extend_name_list = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
coco_name_list = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
name_dict = dict(zip(voc_extend_name_list, coco_name_list))


def coco_classes():
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]


label_ids = {name: i + 1 for i, name in enumerate(coco_classes())}


def box_modified(width, height, bbox: list, pad_size=800):
    big = max(width, height)
    x1, y1, x2, y2 = bbox
    # Normalize
    xc_n = (x1 + x2 + big - width) / (2. * big)
    yc_n = (y1 + y2 + big - height) / (2. * big)
    ww_n = max((x2 - x1) / (1.0 * big), 0.)
    hh_n = max((y2 - y1) / (1.0 * big), 0.)
    xmin = (xc_n - ww_n / 2) * pad_size
    ymin = (yc_n - hh_n / 2) * pad_size
    xmax = (xc_n + ww_n / 2) * pad_size
    ymax = (yc_n + hh_n / 2) * pad_size
    return [xmin, ymin, xmax, ymax]


def parse_xml(args, pad_size=800):
    xml_path, img_path = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        new_name = name_dict.get(str(name))
        obj.find('name').text = new_name
        label = label_ids[new_name]
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        bbox = box_modified(width=w, height=h, bbox=bbox, pad_size=pad_size)
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0,))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0,))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    annotation = {
        'filename': img_path,
        'width': pad_size,
        'height': pad_size,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def cvt_annotations(split, pad_size):
    annotations = []
    img_names = split
    xml_paths = [
        osp.join(VOCdevkit_dir, VOC_type, f'Annotations/{img_name}.xml')
        for img_name in img_names
    ]
    img_paths = [
        f'JPEGImages/{img_name}.png' for img_name in img_names
    ]
    parse_xml_ft = ft.partial(parse_xml, pad_size=pad_size)
    part_annotations = mmcv.track_progress(parse_xml_ft,
                                           list(zip(xml_paths, img_paths)))
    annotations.extend(part_annotations)
    return annotations


def cvt_to_coco_json(annotations):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        annotation_item['segmentation'].append(seg)

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    for category_id, name in enumerate(coco_classes()):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id) + 1
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco


def generateImg(imgs_list: list, size=800):
    print("Generating images ...")
    voc_raw_imgs_path = osp.join(VOCdevkit_dir, VOC_type, "JPEGImages/")
    coco_raw_imgs_path = osp.join(CoCo_output_dir, "train2017/JPEGImages")
    os.makedirs(coco_raw_imgs_path, exist_ok=True)
    transform = iaa.Sequential([
        iaa.PadToAspectRatio(1.0, position="center-center").to_deterministic()
    ])

    for image_file in mmcv.track_iter_progress(imgs_list):
        # Transform raw imgs
        raw_image = np.array(Image.open(osp.join(voc_raw_imgs_path, image_file)).convert('RGB'), dtype=np.uint8)
        raw_image_transformed = Image.fromarray(transform(image=raw_image))
        raw_image_transformed = raw_image_transformed.resize((size, size))
        raw_image_transformed.save(osp.join(coco_raw_imgs_path, osp.splitext(image_file)[0] + '.png'))
    print("Generating images done.")


def generateLabel(split, size=800):
    print("Generating labels ...")
    # train, val, trainval and test are same as default
    annotations = cvt_annotations(split, pad_size=size)
    annotations_json = cvt_to_coco_json(annotations)
    for split in ['train', 'val', 'trainval', 'test']:
        dataset_name = 'instances' + '_' + split + '2017'
        out_dir = osp.join(CoCo_output_dir, "annotations")
        os.makedirs(out_dir, exist_ok=True)
        print(f'processing {dataset_name} ...')

        mmcv.dump(annotations_json, osp.join(out_dir, dataset_name + '.json'))
    print("Generating labels done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Covert VOC dataset to COCO type.")
    parser.add_argument('--voc_type', default="VOC2012", help='voc dataset type')
    parser.add_argument('-n', '--num', type=int, default=2000, help='number of examples')
    parser.add_argument('-s', '--img_size', type=int, default=800, help='image size after processing')
    return parser.parse_args()


if __name__ == '__main__':
    print(os.getcwd())
    args = parse_args()
    img_size = args.img_size
    VOC_type = f"{args.voc_type}_{args.num}"
    VOCdevkit_dir = "./data/VOCdevkit"
    CoCo_output_dir = f"./data/Voc12_CoCo_{img_size}_{args.num}"

    os.makedirs(CoCo_output_dir, exist_ok=True)

    # Get the list of images
    split_list = mmcv.list_from_file(osp.join(VOCdevkit_dir, VOC_type, "ImageSets/Main/trainval.txt"))

    # Generate imgs
    generateImg(imgs_list=[number + ".jpg" for number in split_list], size=img_size)

    # Generate labels
    generateLabel(split_list, size=img_size)

