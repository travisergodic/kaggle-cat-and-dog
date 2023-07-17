
python tools/inference.py --config_file configs/resnet50/resnet_cfg.py --bs 128 \
                          --img_dir ../dogs-vs-cats/test1 \
                          --weight checkpoints/resnet50_v10/best.pt \
                          --save_path result.csv