#!/usr/bin/env sh

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : run.sh
#   Author       : Yufeng Liu
#   Date         : 2021-04-09
#   Description  : 
#
#================================================================

exp_folder="exps/exp002"
mkdir -p $exp_folder
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --deterministic --max_epochs 50 --save_folder ${exp_folder} --amp > ${exp_folder}/fullsize_adam.log &
