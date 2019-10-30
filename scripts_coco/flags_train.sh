#!/usr/bin/env bash
FLAG_PATH=/data/nganltp/data/flags
RESULT_PATH=/data/nganltp/result/flags

python train_setops_stripped_new.py --resume_path="" --inception_transform_input=False --init_inception=False --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=2 --base_network_name=resnet50 --crop_size=224 --epochs=10 --train_base=True --train_classifier=True --coco_path=${FLAG_PATH} --results_path=${RESULT_PATH}