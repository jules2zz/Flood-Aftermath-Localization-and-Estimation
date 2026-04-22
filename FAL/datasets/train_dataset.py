
import os
import torch
import random
import logging
import numpy as np
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import datasets.dataset_utils as dataset_utils

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset_folder, M=10, alpha=30, N=5, L=2,
                 current_group=0, min_images_per_class=10):
        super().__init__()
        self.M = M
        self.alpha = alpha
        self.N = N
        self.L = L
        self.current_group = current_group
        self.dataset_folder = dataset_folder
        self.augmentation_device = args.augmentation_device
        self.image_size = args.image_size

        dataset_name = os.path.basename(dataset_folder)
        filename = f"cache/{dataset_name}_M{M}_N{N}_alpha{alpha}_L{L}_mipc{min_images_per_class}.torch"
        if not os.path.exists(filename):
            os.makedirs("cache", exist_ok=True)
            logging.info(f"Cached dataset {filename} does not exist, I'll create it now.")
            self.initialize(dataset_folder, M, N, alpha, L, min_images_per_class, filename)
        elif current_group == 0:
            logging.info(f"Using cached dataset {filename}")

        classes_per_group, self.images_per_class = torch.load(filename)
        if current_group >= len(classes_per_group):
            raise ValueError(
                f"With this configuration there are only {len(classes_per_group)} groups, therefore I can't create the {current_group}th group."
            )
        self.classes_ids = classes_per_group[current_group]

        if self.augmentation_device == "cpu":
            self.transform = T.Compose([
                T.ColorJitter(brightness=args.brightness, contrast=args.contrast, saturation=args.saturation, hue=args.hue),
                T.RandomResizedCrop([args.image_size, args.image_size], scale=[1 - args.random_resized_crop, 1], antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    @staticmethod
    def open_image(path):
        return Image.open(path).convert("RGB")

    def __getitem__(self, class_num):
        class_id = self.classes_ids[class_num]
        image_path = os.path.join(self.dataset_folder, random.choice(self.images_per_class[class_id]))
        pil_image = TrainDataset.open_image(image_path)
        tensor_image = TF.to_tensor(pil_image)
        expected = torch.Size([3, self.image_size, self.image_size])
        assert tensor_image.shape == expected, f"Image {image_path} should have shape {expected} but has {tensor_image.shape}."
        if self.augmentation_device == "cpu":
            tensor_image = self.transform(tensor_image)
        return tensor_image, class_num, image_path

    def get_images_num(self):
        return sum([len(self.images_per_class[c]) for c in self.classes_ids])

    def __len__(self):
        return len(self.classes_ids)

    @staticmethod
    def initialize(dataset_folder, M, N, alpha, L, min_images_per_class, filename):
        logging.debug(f"Searching training images in {dataset_folder}")
        images_paths = dataset_utils.read_images_paths(dataset_folder)
        logging.debug(f"Found {len(images_paths)} images")
        logging.debug("For each image, get its UTM east, UTM north and heading from its path")
        images_metadatas = [p.split("@") for p in images_paths]
        utmeast_utmnorth_heading = [(m[1], m[2], m[9]) for m in images_metadatas]
        utmeast_utmnorth_heading = np.array(utmeast_utmnorth_heading).astype(np.float64)
        logging.debug("For each image, get class and group to which it belongs")
        class_id__group_id = [TrainDataset.get__class_id__group_id(*m, M, alpha, N, L) for m in utmeast_utmnorth_heading]
        logging.debug("Group together images belonging to the same class")
        images_per_class = {}
        for image_path, (class_id, _) in zip(images_paths, class_id__group_id):
            if class_id not in images_per_class:
                images_per_class[class_id] = []
            images_per_class[class_id].append(image_path)
        images_per_class = {k: v for k, v in images_per_class.items() if len(v) >= min_images_per_class}
        class_id__group_id = [c for c in class_id__group_id if c[0] in images_per_class]
        classes_per_group = [[] for _ in range(max(g for _, g in class_id__group_id) + 1)]
        for class_id, group_id in class_id__group_id:
            if class_id not in classes_per_group[group_id]:
                classes_per_group[group_id].append(class_id)
        classes_per_group = [g for g in classes_per_group if len(g) > 0]
        torch.save((classes_per_group, images_per_class), filename)

    @staticmethod
    def get__class_id__group_id(utm_east, utm_north, heading, M, alpha, N, L):
        rounded_utm_east = int(utm_east // M * M)
        rounded_utm_north = int(utm_north // M * M)
        rounded_heading = int(heading // alpha * alpha)
        class_id = (rounded_utm_east, rounded_utm_north, rounded_heading)
        group_id = (rounded_utm_east % (M * N) // M) + (rounded_utm_north % (M * N) // M) * N + (rounded_heading % (alpha * L) // alpha) * N * N
        return class_id, group_id


class OnlinePairTrainDataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset_folder, M=10, alpha=30, N=5, L=2, current_group=0, min_images_per_class=10):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.image_size = args.image_size
        self.M = M
        self.alpha = alpha
        self.N = N
        self.L = L
        self.current_group = current_group

        dataset_name = os.path.basename(dataset_folder)
        filename = f"cache/{dataset_name}_M{M}_N{N}_alpha{alpha}_L{L}_mipc{min_images_per_class}.torch"
        if not os.path.exists(filename):
            os.makedirs("cache", exist_ok=True)
            logging.info(f"Cached dataset {filename} does not exist, I'll create it now.")
            TrainDataset.initialize(dataset_folder, M, N, alpha, L, min_images_per_class, filename)
        elif current_group == 0:
            logging.info(f"Using cached dataset {filename}")

        classes_per_group, images_per_class = torch.load(filename)
        if current_group >= len(classes_per_group):
            raise ValueError(f"With this configuration there are only {len(classes_per_group)} groups, therefore I can't create the {current_group}th group.")
        self.classes_ids = classes_per_group[current_group]
        self.images_per_class = {cid: images_per_class[cid] for cid in self.classes_ids}

    @staticmethod
    def open_image(path):
        return Image.open(path).convert("RGB")

    def _to_tensor(self, image_path: str) -> torch.Tensor:
        pil_image = self.open_image(image_path)
        tensor_image = TF.to_tensor(pil_image)
        expected = torch.Size([3, self.image_size, self.image_size])
        assert tensor_image.shape == expected, f"Image {image_path} should have shape {expected} but has {tensor_image.shape}."
        return tensor_image

    def sample_raw_image(self, class_num: int):
        class_id = self.classes_ids[class_num]
        rel_path = random.choice(self.images_per_class[class_id])
        image_path = os.path.join(self.dataset_folder, rel_path)
        return self._to_tensor(image_path), class_num, image_path

    def sample_two_raw_images(self, class_num: int):
        class_id = self.classes_ids[class_num]
        candidates = self.images_per_class[class_id]
        if len(candidates) == 1:
            rel1 = rel2 = candidates[0]
        else:
            rel1, rel2 = random.sample(candidates, 2)
        path1 = os.path.join(self.dataset_folder, rel1)
        path2 = os.path.join(self.dataset_folder, rel2)
        return self._to_tensor(path1), self._to_tensor(path2), class_num, path1, path2

    def get_images_num(self):
        return sum([len(self.images_per_class[c]) for c in self.classes_ids])

    def __len__(self):
        return len(self.classes_ids)
