#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : benchmark.py
#   Author       : Yufeng Liu
#   Date         : 2021-04-01
#   Description  : 
#
#================================================================

import numpy as np
import pickle
import time
import os
import glob

from swc_handler import parse_swc


def load_file(f, file_ext):
    if file_ext == 'swc':
        _ = parse_swc(f)
    elif file_ext == 'npy':
        _ = np.load(f)
    elif file_ext == 'npz':
        _ = np.load(f)
        data = _['data']
    elif file_ext == 'pkl':
        with open(f, 'rb') as fp:
            _ = pickle.load(fp)
    else:
        raise NotImplementedError(f"Loading of file type {file_ext} is not supported yet!")

def benchmark(folder, file_ext='npz'):
    # load all files
    files = sorted(glob.glob(os.path.join(folder, f'*.{file_ext}')))
    print(len(files))
    
    nskip = 10
    for i, f in enumerate(files):
        if i == nskip:
            t0 = time.time()
        # exectue load
        load_file(f, file_ext)
    
    tot_t = time.time() - t0
    print(f'Total and average time for file type {file_ext} are: {tot_t:f}s and {tot_t/(len(files)-nskip):f}\n')


if __name__ == '__main__':
    folder = '../../data/task0001_17302'
    for file_ext in ['npy', 'npz', 'swc', 'pkl']:
        benchmark(folder, file_ext)




