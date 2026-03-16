import h5py
import json
import os
import PIL.Image
import argparse

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict


def create_data_lists(split='train', coco_dir="coco", subj=1, coco_annotation_dir="coco/annotations", fmri_dir="fmri_data"):
    fmri_list, img_list, caption_list, coco_id_list = [], [], [], []

    fmri_data = FmriData(split_val=True, subj=subj, fmri_dir=fmri_dir)
    split_data = fmri_data.get_split(split)
    # Load captions only for train/val splits
    if split in ['train', 'val']:
        annotations = json.load(open(f"{coco_annotation_dir}/captions_train2017.json"))['annotations'] + \
                      json.load(open(f"{coco_annotation_dir}/captions_val2017.json"))['annotations']

        for split_temp in ['train', 'val']:
            caption_path = f"{coco_annotation_dir}/captions_{split_temp}2017.json"
            image_to_caption = {ann['image_id']: ann['caption'] for ann in annotations}

    for subject_id, subject_data in split_data.items():
        idx = 0
        for sample_id, sample in subject_data.items():
            try:
                coco_id = sample['coco_id'][()].item()
                fmri = sample['visual_voxels'][()]
                image = load_coco_image(coco_id, coco_dir)
                # plt.imshow(image)
                # plt.axis("off")  # Turn off the axis
                # plt.show()
                if idx % 500 == 0:
                    print(f"Processing sample {idx} of {len(subject_data)}")
                idx += 1
                fmri_list.append(fmri)
                img_list.append(image)
                coco_id_list.append(coco_id)

                if split in ['train', 'val']:
                    caption = image_to_caption.get(coco_id)
                    if caption:
                        caption_list.append(caption)
                    else:
                        # Remove the corresponding entries if no caption found
                        print("Caption NOT found for coco id: ", coco_id)
                        fmri_list.pop()
                        img_list.pop()
                        coco_id_list.pop()

            except Exception as e:
                print(f"Error processing sample {sample_id}: {str(e)}")
                continue

    return fmri_list, img_list, caption_list, coco_id_list


def train_val_split_by_coco_id(data):
    cache_path = "split/default.json"
    coco_to_sample_id = defaultdict(set)
    for key in data:
        coco_id = data[key]['coco_id'][()].item()
        coco_to_sample_id[coco_id].add(key)
    split = json.load(open(cache_path))

    train_keys = split['train']
    val_keys = split['val']
    assert len(coco_to_sample_id) == len(train_keys) + len(val_keys)
    train = {}
    val = {}
    for coco_id in val_keys:
        sample_ids = coco_to_sample_id[coco_id]
        val.update({k: data[k] for k in sample_ids})
    for coco_id in train_keys:
        sample_ids = coco_to_sample_id[coco_id]
        train.update({k: data[k] for k in sample_ids})
    return train, val

def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")

def load_coco_image(image_id, coco_dir):
    for split in ['train', 'val', 'test']:
        path = get_coco_path(split, image_id, coco_dir)
        if os.path.exists(path):
            with PIL.Image.open(path) as img:
                return img.copy()
    raise FileNotFoundError


class FmriData:
    def __init__(self, subj=1, split_val=True, fmri_dir="fmri_data"):
        subj = subj

        self._files = {f"subject_{subj}": h5py.File(f"{fmri_dir}/subject_{subj}.h5", 'r')}
        self.file = {
            'train': { f'subject_{subj}': self._files[f'subject_{subj}']['train']},
            'val': {},
            'test': { f'subject_{subj}': self._files[f'subject_{subj}']['test']},
        }
        if split_val:
            for subject_i in self.file['train']:
                train, val = train_val_split_by_coco_id(self.file['train'][subject_i])
                self.file['train'][subject_i] = train
                self.file['val'][subject_i] = val


    def get_split(self, split):
        return self.file[split]


def save_data_lists(fmri_list, img_list, caption_list, coco_id_list, split, save_dir):
    # Convert images to numpy arrays
    img_arrays = [np.array(img) for img in img_list]

    # Save data
    np.savez(f'{save_dir}/data_{split}.npz',
             fmri=np.array(fmri_list),
             images=np.array(img_arrays),
             captions=np.array(caption_list) if caption_list else np.array([]),
             coco_ids=np.array(coco_id_list))



def load_data_lists(split, save_dir):
    data = np.load(f'{save_dir}/data_{split}.npz', allow_pickle=True)
    return (
        data['fmri'],
        [PIL.Image.fromarray(img) for img in data['images']],
        data['captions'].tolist() if len(data['captions']) > 0 else [],
        data['coco_ids'].tolist()
    )

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_split",
    type=str,
    required=True,
    choices=['train', 'val', 'test'],
    help="data split")

parser.add_argument("--coco_dir", type=str)
parser.add_argument("--coco_annotation_dir", type=str)
parser.add_argument("--subject_id", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--fmri_dir", type=str)


data_split = parser.parse_args().data_split
subject_id = parser.parse_args().subject_id
fmri_dir = parser.parse_args().fmri_dir
save_dir = parser.parse_args().save_dir
coco_dir = parser.parse_args().coco_dir
coco_annotation_dir = parser.parse_args().coco_annotation_dir



fmri_list, img_list, caption_list, coco_id_list = create_data_lists(split=data_split, coco_dir=coco_dir,
                                                                                        subj=int(subject_id),
                                                                                        coco_annotation_dir=coco_annotation_dir,
                                                                               fmri_dir=fmri_dir)
if data_split == 'test':
    # Average fMRI volumes over duplicate coco/image ids.
    fmri_per_id = defaultdict(list)
    img_per_id = {}
    for fmri, img, coco_id in zip(fmri_list, img_list, coco_id_list):
        fmri_per_id[coco_id].append(fmri)
        if coco_id not in img_per_id:
            img_per_id[coco_id] = img
    print("Unique test: ", len(fmri_per_id))

    aggregated_fmri, aggregated_img, aggregated_coco_ids = [], [], []
    for coco_id, fmris in fmri_per_id.items():
        stacked = np.stack(fmris, axis=0)
        aggregated_fmri.append(stacked.mean(axis=0))
        aggregated_img.append(img_per_id[coco_id])
        aggregated_coco_ids.append(coco_id)

    fmri_list, img_list, coco_id_list = aggregated_fmri, aggregated_img, aggregated_coco_ids

print(len(fmri_list), len(img_list), len(caption_list), len(coco_id_list))

save_data_lists(fmri_list, img_list, caption_list, coco_id_list, split=data_split, save_dir=save_dir)

# train 24300
# val 2700
# unique test 1000/test 3000
