import h5py
import numpy as np
from tqdm import tqdm
import os
import cv2
from math import ceil, floor


class Rect:
    def __init__(self, left, top, right, bottom) -> None:
        self._pos = [int(left), int(top), int(right), int(bottom)]

    def __getitem__(self, b):
        return self._pos[b]

    def __str__(self) -> str:
        return str(self._pos)

    def merge(self, rect):
        if isinstance(rect, Rect):
            self._pos[0] = min(self._pos[0], rect[0])
            self._pos[1] = min(self._pos[1], rect[1])
            self._pos[2] = max(self._pos[2], rect[2])
            self._pos[3] = max(self._pos[3], rect[3])
        else:
            self._pos[0] = min(self._pos[0], rect[0])
            self._pos[1] = min(self._pos[1], rect[1])
            self._pos[2] = max(self._pos[2], rect[0])
            self._pos[3] = max(self._pos[3], rect[1])

    def int_rect(self):
        return Rect(floor(self._pos[0]), floor(self._pos[1]), ceil(self._pos[2]), ceil(self._pos[3]))

    def center(self):
        return (np.array(self._pos[:2]) + np.array(self._pos[-2:])) * 0.5

    def add_x(self, delt_x):
        self._pos[0] += delt_x
        self._pos[2] += delt_x

    def add_y(self, delt_y):
        self._pos[1] += delt_y
        self._pos[3] += delt_y

    def fix_in_rect(self, limit_rect):
        self.add_x(max(limit_rect.left - self.left, 0))
        self.add_y(max(limit_rect.top - self.top, 0))
        self.add_x(-max(self.right - limit_rect.right, 0))
        self.add_y(-max(self.bottom - limit_rect.bottom, 0))

    @property
    def left(self):
        return self._pos[0]

    @property
    def top(self):
        return self._pos[1]

    @property
    def right(self):
        return self._pos[2]

    @property
    def bottom(self):
        return self._pos[3]

    @property
    def w(self):
        return self._pos[2] - self._pos[0]

    @property
    def h(self):
        return self._pos[3] - self._pos[1]


data_path = '/data/human/human'
out_dir = '/data/human_64x64/human_trace'

file_rect = {}

for dataset in ['valid', 'train']:
    print(f"dataset: {dataset}\n")

    f = h5py.File(f'{dataset}.h5', 'r')
    # read the skeleton positions
    part_center = np.array(f["part"])

    with open(f'{dataset}_images.txt', 'r') as f:
        filelist = f.readlines()
    print(len(filelist), "\n")
    # find largest bounding box
    for file_name, center_poses in tqdm(zip(filelist, part_center)):
        file_name = file_name.strip()
        file_rect[file_name] = Rect(*list(center_poses[0]), *list(center_poses[0]))
        for center_pos in center_poses[1:]:
            file_rect[file_name].merge(center_pos)
        file_rect[file_name] = file_rect[file_name].int_rect()

PADDING = 200
ORIGINAL_IMAGE_SIZE = 1000
OUTPUT_IMAGE_SIZE = 64

t = {}
# perform padding and resize
for file, rect in tqdm(file_rect.items()):
    d = min(max(rect.w, rect.h) + PADDING, ORIGINAL_IMAGE_SIZE)
    center = rect.center()
    rect = Rect(*(center - np.array([d, d]) / 2), *(center + np.array([d, d]) / 2))
    rect.fix_in_rect(Rect(0, 0, ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE))
    t[file] = rect
file_rect = t

# write images
for file, _ in tqdm(file_rect.items()):
    dir_path = '/data/human/human'
    image = cv2.imread(os.path.join(dir_path, file))
    rect = file_rect[file]
    image = image[rect.top:rect.bottom, rect.left:rect.right]
    cv2.imwrite(os.path.join(out_dir, os.path.basename(file)),
                cv2.resize(image, [OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE]))