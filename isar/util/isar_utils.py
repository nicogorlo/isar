from time import perf_counter_ns
import itertools
import os
import cv2
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

class performance_measure:

    def __init__(self, name) -> None:
        self.name = name

    def __enter__(self):
        self.start_time = perf_counter_ns()
    
    def __exit__(self, *args):
        self.end_time = perf_counter_ns()
        self.duration = self.end_time - self.start_time
        
        print(f"{self.name} - execution time: {(self.duration)/1000000:.2f} ms")

"""
returns an iterator that can be called for the next image in the folder with next(iterator)
"""
def get_image_it_from_folder(datadir) -> itertools.cycle:

    images = itertools.cycle([image for image in sorted(os.listdir(datadir))])
    
    return images

def copy_data(srcdir: str, dstdir: str):

    for scene in tqdm(sorted(os.listdir(srcdir))):
        print("Scene: ", scene)

        if not os.path.exists(os.path.join(dstdir, scene, 'ImageEmbeddings')):
            os.mkdir(os.path.join(dstdir, scene, 'ImageEmbeddings'))

        for file in sorted(os.listdir(os.path.join(srcdir, scene))):
            shutil.copy2(os.path.join(srcdir, scene, file), os.path.join(dstdir, scene, 'ImageEmbeddings', file))



## visualization:

def semantic_obs_to_img(semantic_obs, semantic_palette):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(semantic_palette.flatten())
    semantic_img.putdata((semantic_obs.flatten()).astype(np.uint8))
    semantic_img = semantic_img.convert('RGB')
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_BGR2RGB)
    return semantic_img

def generate_pastel_color():

    r, g, b = [random.randint(64, 255) for _ in range(3)]
    average = (r + g + b) // 3
    r, g, b = [int((x + average) // 2) for x in (r, g, b)]
    return (r, g, b)