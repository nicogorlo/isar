from time import perf_counter_ns
import itertools
import os
import cv2


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
def get_image_it_from_folder(datadir, fps) -> itertools.cycle:

    images = itertools.cycle([datadir + image for image in sorted(os.listdir(datadir))])
    
    return images