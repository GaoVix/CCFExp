import numbers
import mxnet as mx
import os
import pandas as pd
from src.general_utils.os_utils import get_all_files
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from PIL import Image
from scipy import misc
import random


class ListDatasetWithIndex(Dataset):
    def __init__(self, img_list, subset, transform, target_transform, return_label, deterministic, image_is_saved_with_swapped_B_and_R=False):
        super(ListDatasetWithIndex, self).__init__()

        if subset == '0-all':
            start_index = 0
            end_index = len(img_list)
        else:
            assert len(subset.split('-')) == 2
            start_index = int(float(subset.split('-')[0]) * len(img_list))
            end_index = int(float(subset.split('-')[1]) * len(img_list))
        assert start_index >= 0
        assert end_index <= len(img_list)
        assert start_index < end_index
        self.start_index = start_index
        self.end_index = end_index

        self.img_list = img_list
        self.dummy_labels = np.arange(len(img_list)) % 100
        rows = []
        for idx, (name, label) in enumerate(zip(self.img_list, self.dummy_labels)):
            row = {'idx': idx, 'path': '{}/name.jpg'.format(label), 'label': label}
            rows.append(row)
        self.record_info = pd.DataFrame(rows)

        self.transform = transforms
        self.image_is_saved_with_swapped_B_and_R = image_is_saved_with_swapped_B_and_R

        if isinstance(transform, list):
            # split transform for returning both 112x112 and 128x128
            transform_random1, transform_random2, transform_determ1, transform_determ2 = transform
            self.transform_random1 = transform_random1
            self.transform_random2 = transform_random2
            self.transform_determ1 = transform_determ1
            self.transform_determ2 = transform_determ2
            self.split_transform = True
        else:
            self.transform = transform
            self.split_transform = False
        self.target_transform = target_transform
        assert self.target_transform is None  # not implemented yet
        self.return_label = return_label
        self.deterministic = deterministic

        self.rec_label_to_another_label = dict(zip(self.dummy_labels, self.dummy_labels))

    def __len__(self):
        if hasattr(self, 'end_index') and hasattr(self, 'start_index'):
            return self.end_index - self.start_index
        else:
            return len(self.img_list)

    def transform_images(self, sample):
        if self.split_transform:
            if self.deterministic:
                pass
            else:
                sample = self.transform_random1(sample)
            sample1 = self.transform_determ1(sample)

            # sample2 is usually original shape
            if self.deterministic:
                sample2 = sample
            else:
                sample2 = self.transform_random2(sample)
            sample2 = self.transform_determ2(sample2)

        else:
            raise ValueError('not implemented')

        return sample1, sample2

    def read_image(self, idx):

        if self.image_is_saved_with_swapped_B_and_R:
            with open(self.img_list[idx], 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
        else:
            img = cv2.imread(self.img_list[idx])
            if img is None:
                print(self.img_list[idx])
                raise ValueError(self.img_list[idx])
            img = img[:,:,:3]
            img = Image.fromarray(img)
        return img

    def __getitem__(self, index):
        index = index + self.start_index
        sample = self.read_image(index)
        sample, orig_sample = self.transform_images(sample)
        return_dict = {}
        return_dict['image'] = sample
        return_dict['index'] = index
        return_dict['orig'] = orig_sample

        if self.return_label:
            return_dict['class_label'] = torch.tensor(0)
            return_dict['human_label'] = 'subject_0'

        return return_dict

class LoadPalsyData(Dataset):
    def __init__(self,
                 data_root,
                 return_label=True,
                 subset='0-all',
                 image_is_saved_with_swapped_B_and_R=False,
                 return_identity_image='',
                 deterministic=False,
                 target_transform=None,
                 transform=None,
                 rec_label_to_another_label=None,
                 return_extra_same_label_samples=False):
        super(LoadPalsyData, self).__init__()
        self.data_root = data_root
        self.return_label = return_label
        self.deterministic = deterministic
        self.image_is_saved_with_swapped_B_and_R = image_is_saved_with_swapped_B_and_R
        self.return_identity_image = return_identity_image
        self.return_extra_same_label_samples = return_extra_same_label_samples

        self.img_list, self.labels, self.identity_images = self.generate_img_list_and_labels()

        rows = []
        for idx, (name, label) in enumerate(zip(self.img_list, self.labels)):
            row = {'idx': idx, 'path': '{}/name.jpg'.format(label), 'label': label}
            rows.append(row)
        self.record_info = pd.DataFrame(rows)

        if rec_label_to_another_label is None:
            # make one using path
            # image folder with 0/1.jpg

            # from record file label to folder name
            rec_label = self.record_info.label.tolist()
            foldernames = self.record_info.path.apply(lambda x: x.split('/')[0]).tolist()
            self.rec_to_folder = {}
            for i, j in zip(rec_label, foldernames):
                self.rec_to_folder[i] = j

            # from folder name to number as torch imagefolder
            foldernames = sorted(str(entry) for entry in self.rec_to_folder.values())
            self.folder_to_num = {cls_name: i for i, cls_name in enumerate(foldernames)}
            self.rec_label_to_another_label = {}

            # combine all
            for x in rec_label:
                self.rec_label_to_another_label[x] = self.folder_to_num[self.rec_to_folder[x]]

        else:
            self.rec_label_to_another_label = rec_label_to_another_label

        if isinstance(transform, list):
            # split transform for returning both 112x112 and 128x128
            transform_random1, transform_random2, transform_determ1, transform_determ2 = transform
            self.transform_random1 = transform_random1
            self.transform_random2 = transform_random2
            self.transform_determ1 = transform_determ1
            self.transform_determ2 = transform_determ2
            self.split_transform = True
        else:
            self.transform = transform
            self.split_transform = False
        self.target_transform = target_transform
        assert self.target_transform is None  # not implemented yet

        if subset == '0-all':
            self.start_index = 0
            self.end_index = len(self.img_list)
        else:
            start, end = map(float, subset.split('-'))
            self.start_index = int(start * len(self.img_list))
            self.end_index = int(end * len(self.img_list))

        if self.return_extra_same_label_samples:
            self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0] for label in set(self.labels)}

    def generate_img_list_and_labels(self):
        img_list = []
        labels = []
        identity_images = {}
        label_to_idx = {}
        idx = 0

        for root, dirs, files in os.walk(self.data_root):
            if not files:  # Skip empty directories
                continue
            label = os.path.basename(root)


            if label not in label_to_idx:

                label_to_idx[label] = idx

                idx += 1

            identity_found = False

            for file in files:
                if file.endswith((".png", ".jpg", ".jpeg")):
                    if not identity_found and self.return_identity_image :
                        identity_images[label_to_idx[label]] = os.path.join(root, file)
                        identity_found = True

                    path = os.path.join(root, file)
                    img_list.append(path)
                    labels.append(label_to_idx[label])

        return img_list, labels, identity_images

    def __len__(self):
        return self.end_index - self.start_index

    def transform_images(self, sample):
        if self.split_transform:
            if self.deterministic:
                pass
            else:
                sample = self.transform_random1(sample)
            sample1 = self.transform_determ1(sample)

            # sample2 is usually original shape
            if self.deterministic:
                sample2 = sample
            else:
                sample2 = self.transform_random2(sample)
            sample2 = self.transform_determ2(sample2)

        else:
            raise ValueError('not implemented')

        return sample1, sample2

    def read_image(self, path, convert_to_rgb=True):
        if self.image_is_saved_with_swapped_B_and_R or not convert_to_rgb:
            img = Image.open(path)
            img = img.convert('RGB')
        else:
            img = Image.open(path).convert('RGB')
        return img

    def __getitem__(self, idx):
        idx += self.start_index
        img_path = self.img_list[idx]
        img = self.read_image(img_path)
        label = self.labels[idx]

        img, orig = self.transform_images(img)
        result = {'orig': orig, 'index': idx, 'class_label': torch.tensor(label, dtype=torch.long), 'image': img}

        if self.return_identity_image:
            id_img_path = self.identity_images[self.labels[idx]]
            id_img = self.read_image(id_img_path)
            id_img, orig_id_image = self.transform_images(id_img)
            result['id_image'] = id_img

        if self.return_extra_same_label_samples:
            same_label_idx = idx
            while same_label_idx == idx:
                same_label_idx = random.choice(self.label_to_indices[label])
            extra_img_path = self.img_list[same_label_idx]
            extra_img = self.read_image(extra_img_path)
            extra_img, orig_extra_img = self.transform_images(extra_img)
            result['extra_image'] = extra_img

        return result


def make_dataset(data_path,
                 deterministic=False,
                 img_size=112,
                 return_extra_same_label_samples=False,
                 subset='0-all',
                 orig_augmentations1=[],
                 orig_augmentations2=[],
                 return_identity_image=''):


    transform_random1 = []
    if not deterministic and orig_augmentations1:
        for aug in orig_augmentations1:
            prob = float(aug.split(":")[-1])
            if 'flip' in aug:
                t = transforms.RandomApply(transforms=[transforms.RandomHorizontalFlip()], p=prob)
            elif 'gray' in aug:
                t = transforms.RandomApply(transforms=[transforms.Grayscale(num_output_channels=3)], p=prob)
            elif 'photo' in aug:
                t = transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=.3, contrast=.3)], p=prob)
            else:
                raise ValueError('not correct')
            transform_random1.append(t)
    transform_random1 = transforms.Compose(transform_random1)

    transform_random2 = []
    if not deterministic and orig_augmentations2:
        for aug in orig_augmentations2:
            prob = float(aug.split(":")[-1])
            if 'flip' in aug:
                t = transforms.RandomHorizontalFlip()
            elif 'gray' in aug:
                t = transforms.RandomApply(transforms=[transforms.Grayscale(num_output_channels=3)], p=prob)
            elif 'photo' in aug:
                t = transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=.3, contrast=.3)], p=prob)
            else:
                raise ValueError('not correct')
            transform_random2.append(t)
    transform_random2 = transforms.Compose(transform_random2)

    transform_determ1 = [transforms.Resize(img_size),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform_determ1 = transforms.Compose(transform_determ1)
    transform_determ2 = [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform_determ2 = transforms.Compose(transform_determ2)

    if 'palsy_f' in data_path:
        dataset = LoadPalsyData(data_path,
                                return_label=True,
                                subset=subset,
                                rec_label_to_another_label=None,
                                transform=[transform_random1, transform_random2, transform_determ1, transform_determ2],
                                target_transform=None,
                                return_extra_same_label_samples=return_extra_same_label_samples,
                                deterministic=deterministic,
                                return_identity_image=return_identity_image,
                                image_is_saved_with_swapped_B_and_R=False,
                                )

        return dataset


