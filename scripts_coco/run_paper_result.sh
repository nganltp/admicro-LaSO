LASO_MODEL=/data/nganltp/LaSOmodels
COCO_PATH=/data/nganltp/data/coco_2014
RESULT_PATH=/data/nganltp/result/coco_2014

python test_precision.py --unseen=False --resume_path=$LASO_MODEL --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=1 --resume_epoch=4 --base_network_name=Inception3 --init_inception=True --crop_size=299 --skip_tests=1 --paper_reproduce=True --coco_path=$COCO_PATH --results_path=$RESULT_PATH
