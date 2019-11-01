#!/usr/bin/env bash
FLAG_PATH=/data/nganltp/data/flags
RESULT_PATH=/data/nganltp/result/flags

python test_precision_flag.py --unseen=False --resume_path=/tmp/results/train_setops_stripped_new/0016_73b06f3/noJobID/191030_153310 --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=2 --resume_epoch=47 --base_network_name=resnet50 --init_inception=False --crop_size=224 --skip_tests=1 --avgpool_kernel=10 --coco_path=/home/nganltp/laso/data/flags --results_path=/data/nganltp/result/flags --num_class=7