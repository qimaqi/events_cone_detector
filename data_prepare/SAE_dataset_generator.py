# -*- coding: utf-8 -*-


import os
import numpy as np
from matplotlib import pyplot as plt
import cv2

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

plt.close('all')
from PIL import Image
from tqdm import tqdm

# %% Read a file of events and write another file with a subset of them
#filename_sub = 'slider_depth/events_chunk.txt'
filename_sub = './../dataset/events.txt'

"""
# This is how the file events_chunk.txt was generated from the events.txt file in the IJRR 2017 dataset
events_raw = open('slider_depth/events.txt', "r")
events_sub = open(filename_sub, "w")
# format: timestamp, x, y, polarity

for k in range(50000):
    line = events_raw.readline()
    #print(line)
    events_sub.write(line)
    
events_raw.close()
events_sub.close()
"""



# test_list = [0,1,2,3,4]
# t_test = 2
# print("find_cloest_time", find_cloest_time(test_list, t_test))
# %% Sensor size

# Get the size of the sensor using a grayscale frame (in case of a DAVIS)
# filename_frame = 'slider_depth/images/frame_00000000.png'
# import cv2
# img = cv2.imread(filename_frame, cv2.IMREAD_GRAYSCALE)
# print img.shape
# img = np.zeros(img.shape, np.int)

# For this exercise, we just provide the sensor size (height, width)
# img_size = (180,240)
img_size = (1080,1920)
images_num = 2990 # 2990 image get 2989 SAE image
fps = 60
SAE_num = images_num-1
timestamps_sae = np.arange(start=1, stop = 2990)/fps # sum
tau = 1/60
save_name_list = sorted(os.listdir('/home/qimaqi/AMZ/Events_yolov5/runs/detect/exp/labels'))
events_npz_path = '/home/qimaqi/AMZ/Events/dataset/amz_events_result/seq1'
events_npz_list = sorted(os.listdir(events_npz_path))

last_index = 0
save_folder = '/home/qimaqi/AMZ/Events/dataset/sae_detect'
save_folder_map = '/home/qimaqi/AMZ/Events/dataset/sae_draw'
assert len(save_name_list)-1 == len(timestamps_sae),"size mismatch"


# img_pos = np.zeros(img_size, np.int)
# img_neg = np.zeros(img_size, np.int)
# for i in range(num_events):
#     if (pol[i] > 0):
#         img_pos[y[i],x[i]] += 1 # count events
#     else:
#         img_neg[y[i],x[i]] += 1

# for detectioin 3000 images
# we need 30
for index,timestamp_sae in tqdm(enumerate(timestamps_sae),total=len(timestamps_sae)):
    
    if index % 100 == 0:
        save_name = save_name_list[index+1].split('.')[0]
        # print("save name",save_name,"timestamp_sae",timestamp_sae)

        events_npz_file_i = os.path.join(events_npz_path, events_npz_list[index])
        # print("events_npz_path, events_npz_list[index]",events_npz_path, events_npz_list[index])
        events_dict_i = np.load(events_npz_file_i)#.files()

        timestamp = events_dict_i['t']/1e9
        x = events_dict_i['x']
        y = events_dict_i['y']

        img_i = np.zeros(img_size, np.float32)
        # print("timestamp events",timestamp[-1])
        # nearest_index = find_cloest_time(timestamp, timestamp_sae)
        # use events between
        # load events npz


        for i in range(len(timestamp)):
            img_i[y[i],x[i]] = np.exp(-(timestamp_sae-timestamp[i]) / tau)

        fig = plt.figure()
        fig.suptitle('Time surface (exp decay). Both polarities')
        plt.imshow(img_i)
        plt.xlabel("x [pixels]")
        plt.ylabel("y [pixels]")
        # plt.colorbar()
        plt.savefig(os.path.join(save_folder_map,save_name+'.png'))
        # plt.show()

        # # change to rgb image   
        img_i = np.expand_dims(img_i,axis=2)
        img_i = img_i/(img_i.max()+1e-6)
        img_rgb = np.repeat(img_i,3,axis=2)

        # fig = plt.figure()
        # fig.suptitle('Time surface rgb')
        # plt.imshow(img_rgb)
        # plt.xlabel("x [pixels]")
        # plt.ylabel("y [pixels]")
        # plt.show()

        # save rgb
        img_rgb_save = Image.fromarray(np.uint8(img_rgb*255))
        img_rgb_save.save(os.path.join(save_folder,save_name+'.png'))


