#!/usr/bin/env sh

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : ddp.sh
#   Author       : Yufeng Liu
#   Date         : 2021-04-10
#   Description  : 
#
#================================================================

MODEL=exps/exp040/final_model.state_dict

CUDA_VISIBLE_DEVICES=0 \
python -u \
    zmq_inference.py \
    --deterministic \
    --amp \
    --image_shape 128,128,128 \
    --enhancement DTGT \
    --deterministic \
    --checkpoint ${MODEL}
    

