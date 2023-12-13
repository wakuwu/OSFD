import os
import shutil
import argparse
import mmcv
import numpy as np
import os.path as osp

random_seed = 2023


def parse_args():
    parser = argparse.ArgumentParser(description="Construct a subset for attack.")
    parser.add_argument('--voc_type', default="VOC2012", help='voc dataset type')
    parser.add_argument('-n', '--num', type=int, default=2000, help='number of examples')
    parser.add_argument('--paper', type=bool, default=False, help='similar dataset for paper')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    VOC_subset_num = args.num
    VOCdevkit_dir = "./data/VOCdevkit"
    VOC_type = args.voc_type
    VOC_subset_path = VOCdevkit_dir
    VOC_subset_type = f"{VOC_type}_{VOC_subset_num}"

    if args.paper:
        subset_index = mmcv.list_from_file("data/image_index.txt")
    else:
        # load split list
        split_list = np.array(mmcv.list_from_file(osp.join(VOCdevkit_dir, VOC_type, "ImageSets/Main/trainval.txt")))
        # construct the subset
        np.random.seed(random_seed)
        shuffled_index = np.random.permutation(len(split_list))
        subset_index = split_list[shuffled_index[0: VOC_subset_num]].tolist()
    subset_index.sort()
    #
    split_save_path = osp.join(VOC_subset_path, VOC_subset_type, "ImageSets/Main/")
    os.makedirs(split_save_path, exist_ok=True)
    with open(osp.join(split_save_path, "trainval.txt"), 'w') as f:
        for idx in subset_index:
            f.writelines(idx+"\n")
    # copy images and labels
    image_save_path = osp.join(VOC_subset_path, VOC_subset_type, "JPEGImages/")
    label_save_path = osp.join(VOC_subset_path, VOC_subset_type, "Annotations/")
    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(label_save_path, exist_ok=True)
    for idx in mmcv.track_iter_progress(subset_index):
        image_name = idx + ".jpg"
        shutil.copyfile(osp.join(VOCdevkit_dir, VOC_type, "JPEGImages/", image_name),
                        osp.join(image_save_path, image_name))
        label_name = idx + ".xml"
        shutil.copyfile(osp.join(VOCdevkit_dir, VOC_type, "Annotations/", label_name),
                        osp.join(label_save_path, label_name))