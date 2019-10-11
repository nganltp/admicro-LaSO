LASO_MODEL=/data/nganltp/LaSOmodels
COCO_PATH=/data/nganltp/data/coco_2014
RESULT_PATH=/data/nganltp/result/coco_2014

python -m cProfile -o out.profile train_setops_stripped.py --inception_transform_input=False --resume_path=$LASO_MODEL/resnet_base_model_only --resume_epoch=4 --init_inception=False --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=2 --base_network_name=resnet50 --crop_size=224 --epochs=2 --train_base=False --coco_path=$COCO_PATH --results_path=$RESULT_PATH
