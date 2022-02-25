from configs import transforms_config

DATASETS = {
    "afhq": {
        'transforms': afhq_Transforms,
        'train_dir': "../data/afhq/train",
        "val_dir": "../data/afhq/val",
        'seed': 69,
        'labels': ["cat", "dog"],  
    }
}