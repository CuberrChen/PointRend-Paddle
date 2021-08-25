

export CUDA_VISIBLE_DEVICES=0 
python train.py --config configs/pointrendfpn/pointrend_resnet101_os8_cityscapes_512×1024_80k.yml--num_workers 4 --use_vdl --do_eval --save_interval 1000 --save_dir output --batch_size 4