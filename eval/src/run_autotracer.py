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
from fuse_image_with_seg import get_image_brain_mapping

def exec_app2(input_image, output_image, 
              vaa3d="/PBshare/SEU-ALLEN/Users/zuohan/vaa3d/start_vaa3d.sh", 
              plugin="/PBshare/SEU-ALLEN/Users/zuohan/vaa3d/plugins/neuron_tracing/Vaa3D_Neuron2/libvn2.so", 
              marker_file='/media/lyf/storage/seu_mouse/configs/m_512_512_256.marker'):
    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} -x {plugin} -f app2 -i {input_image} -o {output_image} -p {marker_file} 0 AUTO 0'
    try:
        p = subprocess.check_output(cmd_str, timeout=6000, shell=True)
    except subprocess.TimeoutExpired:
        print(f"Execution of image: {input_image} is too time-consuming. Skip!")
        p = ''

    return p

def exec_app2_with_bgThresh(input_image, output_image, thresh,
              vaa3d="/home/lyf/Softwares/installation/Vaa3D/v3d_external/bin/vaa3d", 
              plugin="/home/lyf/Softwares/installation/Vaa3D/v3d_external/bin/plugins/neuron_tracing/Vaa3D_Neuron2/libvn2.so", 
              marker_file='/media/lyf/storage/seu_mouse/configs/m_512_512_256.marker'):
    cmd_str = f"{vaa3d} -x {plugin} -f app2 -i {input_image} -o {output_image} -p {marker_file} 0 {thresh} 0"
    try:
        p = subprocess.check_output(cmd_str, timeout=6000, shell=True)
    except subprocess.TimeoutExpired:
        print(f"Execution of image: {input_image} is too time-consuming. Skip!")
        p = ''

    return p

def exec_rivulet(input_image, vaa3d="/home/lyf/Softwares/installation/Vaa3D/v3d_external/bin/vaa3d"):
    cmd_str = f"{vaa3d} -x Rivulet2 -f tracing_func -i {input_image} -p 1 10 0 1"
    try:
        p = subprocess.check_output(cmd_str, timeout=6000, shell=True)
    except subprocess.TimeoutExpired:
        print('The current rivulet tracing of image is time-consuming, most probably errous. Skip!')
        p = ''

    return p

def exec_smartTrace(input_image, vaa3d="/home/lyf/Softwares/installation/Vaa3D/v3d_external/bin/vaa3d"):
    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} -x smartTrace -f smartTrace -i {input_image} -p 1'
    print(cmd_str)
    try:
        p = subprocess.check_output(cmd_str, timeout=100000, shell=True)
    except subprocess.TimeoutExpired:
        print('The current smarttracing of image is time-consuming, most probably errous. Skip!')
        p = ''

    return p

def exec_brain(input_folder, output_folder, marker_file):
    print('===> Processing for brain: {:s}'.format(input_folder))

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for infile in glob.glob(os.path.join(input_folder, "*v3draw")):
        filename = os.path.split(infile)[-1]
        filename = os.path.splitext(filename)[0]
        print('--> Processing for file: {:s}'.format(filename))
        outfile = os.path.join(output_folder, '{:s}.swc'.format(filename))
        # execute conversion
        exec_app2(infile, outfile, marker_file=marker_file)


if __name__ == "__main__":
    
    # APP2 tracing
    nproc = 4
    tracer = 'smartTrace'
    imgdir = '../exp040_brains2/fused_tg0.0_alpha0.8_vanilla_bgMask0'
    marker_file = '/PBshare/SEU-ALLEN/Users/yfliu/transtation/seu_mouse/configs/m_256_256_128.marker'
    mapper, img_dict = get_image_brain_mapping()
    executable = '/PBshare/SEU-ALLEN/Users/zuohan/vaa3d/start_vaa3d.sh'
    args_list = []
    for imgfile in sorted(glob.glob(os.path.join(imgdir, f'[1-9]*.tiff'))):
        prefix = os.path.split(imgfile)[-1][:-5]
        #if prefix == '9506_1437_2117': continue

        brain_id = mapper[prefix]
        if tracer == 'app2':
            outfile = os.path.join(imgdir, f'{prefix}_{brain_id}.swc')
            args = imgfile, outfile, executable, 'vn2', marker_file
        elif tracer == 'smartTrace':    
            outfile = f'{imgfile}_smartTracing.swc'
            args = imgfile, executable # smartTracing
        if os.path.exists(outfile): continue
        args_list.append(args)

    print(f'Number of images to process: {len(args_list)}')
    pt = Pool(nproc)
    if tracer == 'app2':
        pt.starmap(exec_app2, args_list)
    elif tracer == 'smartTrace':
        pt.starmap(exec_smartTrace, args_list)
    pt.close()
    pt.join()
    
    

   
    """
    # Rivulet tracing
    imgdir = '/media/lyf/storage/seu_mouse/crop_data/processed/dendriteImageSecR/tiff'
    args_list = []
    for brain_dir in glob.glob(os.path.join(imgdir, f'[1-9]*[0-9]')):
        print(brain_dir)
        for imgfile in glob.glob(os.path.join(brain_dir, f'[1-9]*[0-9].tiff')):
            print(imgfile)
            filename = os.path.split(imgfile)[-1]
            outpath = os.path.split(imgfile)[0]
            outfile = os.path.join(outpath, f'{filename}_smartTracing.swc')
            if os.path.exists(outfile): continue
            #if prefix != '2577_6443_2262': continue
            args = imgfile, "/home/lyf/Softwares/installation/Vaa3D/v3d_external/bin/vaa3d"
            args_list.append(args)
    
    print(f'Number of images to processi: {len(args_list)}')
    pt = Pool(14)
    pt.starmap(exec_smartTrace, args_list)
    pt.close()
    pt.join()
    """

    """
    # APP2 tracing with user-defined threshold
    nproc = 4
    bgThresh = 10
    imgdir = '/home/lyf/Temp/exp040/EqHist/fused_tg0.0_alpha0.8_vanilla_bgMask0'
    outdir = './app2_AUTO1.0std'
    orig_imgdir = '/media/lyf/storage/seu_mouse/crop_data/processed/dendriteImageSecR/tiff'
    marker_file = '/media/lyf/storage/seu_mouse/configs/m_256_256_128.marker'
    mapper, img_dict = get_image_brain_mapping()
    executable = '/home/lyf/Softwares/installation/Vaa3D/v3d_external/bin/vaa3d'
    args_list = []
    for imgfile in sorted(glob.glob(os.path.join(imgdir, f'[1-9]*.tiff'))):
        prefix = os.path.split(imgfile)[-1][:-5]

        brain_id = mapper[prefix]
        outfile = os.path.join(outdir, f'{prefix}_{brain_id}.swc')
        
        orig_imgfile = os.path.join(orig_imgdir, f'{brain_id}', f'{prefix}.tiff')
        #outfile = f'{imgfile}_smartTracing.swc'
        if os.path.exists(outfile): continue
        #args = imgfile, executable # smartTracing
        args = orig_imgfile, outfile, executable, 'vn2', marker_file
        args_list.append(args)

    print(f'Number of images to processi: {len(args_list)}')
    pt = Pool(nproc)
    pt.starmap(exec_app2, args_list)
    pt.close()
    pt.join()
    """
