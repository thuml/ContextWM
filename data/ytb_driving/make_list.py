# TODO: add comments

import os
import numpy as np
import cv2

data_path = "./newytb"
target_path = "./ytb64x64"

if not os.path.exists(target_path):
    os.mkdir(target_path)
    
OUTPUT_WIDTH = 64
total_frames = 0
total_dirs = 0

list_file = 'ytb_full.txt'

dir_range = {}

step=5

SKIP_HEAD = 100

with open(list_file, 'w') as f:
    for dir_path, dirs, files in os.walk(data_path):
        if os.path.samefile(data_path, dir_path): continue
        dir_name = os.path.split(dir_path)[-1]
        total_dirs += 1
        total_frames += len(files)
        file_ids = [int(file.split('.')[0]) for file in files]
        print("min", np.min(file_ids), np.max(file_ids), "total", len(files))
        idx = 0
        last_i = -10
        vid_indx = -1
        skip_head = SKIP_HEAD
        for file in sorted(files, key=lambda x: int(x.split('.')[0])):
            i = int(file.split('.')[0])
            if i != last_i + 1:
                vid_indx += 1
                skip_head = SKIP_HEAD
                if idx != 0:
                    tartget_dir_name = dir_name + "_" + str(vid_indx)
                    f.write(f"{os.path.join(target_path, tartget_dir_name)} {idx} None\n")
                    idx = 0
                
            last_i = i
            skip_head -= 1
            if skip_head > 0: continue
            if i%step !=0 : continue
            image = cv2.imread(os.path.join(dir_path, f"{i}.jpg"))
            d = min(*image.shape[:2])
            center = (image.shape[0]//2, image.shape[1]//2)
            image = image[center[0]-d//2:center[0]+d//2, center[1]-d//2:center[1]+d//2]
            image = cv2.resize(image, (64, 64))
            tartget_dir_name = dir_name + "_" + str(vid_indx)
            if not os.path.exists(os.path.join(target_path, tartget_dir_name)):
                os.mkdir(os.path.join(target_path, tartget_dir_name))
                print(os.path.join(target_path, tartget_dir_name))
            cv2.imwrite(os.path.join(target_path, tartget_dir_name, f"{idx}.jpg"), image)
            idx += 1
        tartget_dir_name = dir_name + "_" + str(vid_indx)
        f.write(f"{os.path.join(target_path, tartget_dir_name)} {idx} None\n")
            

print("total_dirs:", total_dirs, " total_frames:", total_frames)