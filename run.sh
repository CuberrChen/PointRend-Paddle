# # 创建cityscape文件夹
mkdir -p data/cityscapes/
# # 解压数据集中的gtFine
unzip  /root/paddlejob/workspace/train_data/datasets/data48855/gtFine_train.zip -d data/cityscapes/gtFine/
unzip  /root/paddlejob/workspace/train_data/datasets/data48855/gtFine_val.zip -d data/cityscapes/gtFine/
unzip  /root/paddlejob/workspace/train_data/datasets/data48855/gtFine_test.zip -d data/cityscapes/gtFine/
# # 解压数据集中的leftImg8bit
unzip  /root/paddlejob/workspace/train_data/datasets/data48855/leftImg8bit_train.zip -d data/cityscapes/leftImg8bit/
unzip  /root/paddlejob/workspace/train_data/datasets/data48855/leftImg8bit_val.zip -d data/cityscapes/leftImg8bit/
unzip  /root/paddlejob/workspace/train_data/datasets/data48855/leftImg8bit_test.zip -d data/cityscapes/leftImg8bit/

python tools/create_dataset_list.py data/cityscapes/ --type cityscapes --separator "," 

pip install -r requirements.txt

export CUDA_VISIBLE_DEVICES=0,1,2,3 

python -m paddle.distributed.launch train.py --config configs/pointrendfpn/pointrend_resnet101_os8_cityscapes_512×1024_80k.yml --num_workers=16 --use_vdl --do_eval --save_interval 8000 --save_dir /root/paddlejob/workspace/output
