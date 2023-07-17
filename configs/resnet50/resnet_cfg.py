num_classes = 2
img_size = (224, 224)
device = "cuda"
num_workers = 2
monitor = "accuracy"
n_epochs = 30
save_freq = 1

model_cfg = {
    "type": "resnet50", "num_classes": num_classes, "pretrain": True, "dropout": 0.3, "checkpoint_path": None
}

optimizer_cfg = {
    "type": "Adam", "get_params": lambda model: model.parameters(), "lr": 1e-4, 
    "betas": (0.9, 0.999), "eps": 1e-08, "amsgrad": True
}

epoch_scheduler_cfg = {'type': None}
iter_scheduler_cfg = {'type': None}  