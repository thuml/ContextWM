import os
from tqdm import tqdm
import numpy as np

data_dir = "/data/human/human"
out_file = "./human_full.txt"

video_file_lists = {}
for dir_path, dirs, files in os.walk(data_dir):
    for file in tqdm(sorted(files)):
        title, info, suffix = str(file).split('.')
        pos_id, frame_id = info.split('_')
        frame_id = int(frame_id)
        title = f"{title}.{pos_id}"
        if video_file_lists.get(title) is None:
            video_file_lists[title] = []
        video_file_lists[title].append(os.path.join(dir_path, file))
        
with open(out_file, "w") as f:
    for title, files in video_file_lists.items():
        lable = title.split('_')[1]
        if '.' in lable:
            lable = lable.split('.')[0]
        f.write(f"{title} {len(files)} {lable}\n")