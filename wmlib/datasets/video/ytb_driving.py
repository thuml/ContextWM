from torch.utils.data import Dataset

from PIL import Image
import os
import os.path
import numpy as np

from .utils import VideoRecord


class YoutubeDriving(Dataset):
    def __init__(self, root_path, list_file, segment_len=50, image_tmpl='{:d}.jpg'):
        self.root_path = root_path
        self.list_file = list_file
        self.segment_len = segment_len
        self.image_tmpl = image_tmpl

        self._parse_list(self.segment_len)

    def _parse_list(self, minlen):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1]) >= minlen]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d' % (len(self.video_list)))

    @property
    def total_steps(self):
        return sum([record.num_frames for record in self.video_list])

    def _image_path(self, directory, index):
        return os.path.join(self.root_path, directory, self.image_tmpl.format(index))

    # Modified
    def _load_image(self, directory, idx):
        image = Image.open(self._image_path(directory, idx)).convert('RGB')  # Modified
        image = image.resize([64, 64])
        return np.array(image)

    def _sample_index(self, record):
        return np.random.randint(0, record.num_frames - self.segment_len + 1)

    def get(self, record, ind):
        images = []
        p = ind
        for i in range(self.segment_len):
            seg_imgs = self._load_image(record.path, p)
            images.append(seg_imgs)
            if p < record.num_frames - 1:
                p += 1
        return np.array(images)

    def __getitem__(self, index):
        record = self.video_list[index]
        while not os.path.exists(self._image_path(record.path, 0)):  # Modified
            print(self._image_path(record.path, 0))  # Modified
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]

        segment_index = self._sample_index(record)
        segment = self.get(record, segment_index)
        return segment

    def __len__(self):
        return len(self.video_list)
