#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : run_app2.py
#   Author       : Yufeng Liu
#   Date         : 2021-03-24
#   Description  : 
#
#================================================================

import os, glob
import subprocess
import numpy as np
from multiprocessing.pool import Pool


def do_imPreProcess(inimg_file, outimg_file, 
                    vaa3d="/PBshare/SEU-ALLEN/Users/zuohan/vaa3d/start_vaa3d.sh", 
                    plugin="imPreProcess"):
    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} -x {plugin} -f im_enhancement -i {inimg_file} -o {outimg_file}'
    try:
        p = subprocess.check_output(cmd_str, timeout=1000, shell=True)
    except subprocess.TimeoutExpired:
        print(f"Execution of image: {input_image} is too time-consuming. Skip!")
        p = ''
    except:
        return ''

    return p

def load_file_set(set_file='../../../data/img_singleSoma.list'):
    fset = []
    with open(set_file) as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line: continue
            fset.append(line)
    fset = set(fset)
    return fset


if __name__ == "__main__":
    
    # APP2 tracing
    nproc = 4
    imgdir = '/PBshare/SEU-ALLEN/Users/yfliu/transtation/seu_mouse/crop_data/dendriteImageSecR/tiff'
    outdir = './enh'
    executable = '/PBshare/SEU-ALLEN/Users/zuohan/vaa3d/start_vaa3d.sh'

    fset = load_file_set()
    args_list = []
    for brain_dir in glob.glob(os.path.join(imgdir, '[1-9]*')):
        for imgfile in glob.glob(os.path.join(brain_dir, '[1-9]*.tiff')):
            fname = os.path.split(imgfile)[-1]
            prefix = os.path.splitext(fname)[0]
            if prefix not in fset: 
                continue

            outfile = os.path.join(outdir, fname)
            args = imgfile, outfile, executable
            if os.path.exists(outfile): continue
            args_list.append(args)

    print(f'Number of images to process: {len(args_list)}')
    pt = Pool(nproc)
    pt.starmap(do_imPreProcess, args_list)
    pt.close()
    pt.join()
    
