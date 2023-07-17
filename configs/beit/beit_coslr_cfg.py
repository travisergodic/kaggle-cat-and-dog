# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/beit.py
num_classes = 2
img_size = (224, 224)
device = "cuda"
num_workers = 2
monitor = "accuracy"
n_epochs = 30
save_freq = 1

model_cfg = {
    "type": "beitv2", "name": "'beitv2_base_patch16_224.in1k_ft_in22k_in1k'", 
    "num_classes": num_classes, "dropout": 0.3, "pretrain": True, "checkpoint_path": None
}

optimizer_cfg = {
    "type": "Adam", "get_params": lambda model: model.parameters(), "lr": 1e-4, 
    "betas": (0.9, 0.999), "eps": 1e-08, "amsgrad": True
}

epoch_scheduler_cfg = {'type': None}
iter_scheduler_cfg = {
    "type": "CosineLR", "t_initial": 30, "cycle_decay": 0.5, "lr_min": 1e-5, 
    't_in_epochs': True, 'warmup_t': 2, "warmup_lr_init": 1e-5, "cycle_limit": 1
}
