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
import argparse

# %% Read a file of events and write another file with a subset of them
#filename_sub = 'slider_depth/events_chunk.txt'
filename_sub = './../dataset/events.txt'

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("txt", help="ROS bag file to extract")
parser.add_argument("--timestamps",  help="Timestamps file path")  # /dvs/events
parser.add_argument("--tau", default=0.3, help="decay parameter")
parser.add_argument("--save_folder", default="/result", help="Depth map topic")

args = parser.parse_args()

img_size = (180,240)
save_folder = args.save_folder

def extract_data(filename):
    infile = open(filename, 'r')
    timestamp = []
    x = []
    y = []
    pol = []
    for line in infile:
        words = line.split()
        if len(words)==4:
            timestamp.append(float(words[0]))
            x.append(int(words[1]))
            y.append(int(words[2]))
            pol.append(int(words[3]))
        else:
            break
    infile.close()
    return timestamp,x,y,pol

def find_closest_time(x,t):
    '''
    x timestamp array
    t the time need to find
    '''
    diff_x = np.array(x) - t
    diff_x = diff_x >= 0
    return np.argmax(diff_x) 

# Call the function to read data    
timestamp, x, y, pol = extract_data(args.txt)

timestamps_sae = np.genfromtxt(args.timestamps)

last_index = 0 # default set 0 at start
for index,timestamp_sae in tqdm(enumerate(timestamps_sae),total=len(timestamps_sae)):
    
    save_name = str(index).zfill(6)
    
    current_index = find_closest_time(timestamp, timestamp_sae)

    timestamp_current = timestamp[last_index:current_index]
    x_current = x[last_index:current_index]
    y_current = y[last_index:current_index]

    img_i = np.zeros(img_size, np.float32)

    for i in range(len(timestamp_current)):
        img_i[y_current[i],x_current[i]] = np.exp(-(timestamp_sae-timestamp_current[i]) / tau)

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

    last_time = current_index
