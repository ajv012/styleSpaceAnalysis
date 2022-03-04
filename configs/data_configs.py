from configs import transforms_config

DATASETS = {
    "afhq": {
        'transforms': transforms_config.afhq_Transforms,
        'train_dir': "/data/vision/polina/scratch/avaidya/data/afhq/train",
        "val_dir": "/data/vision/polina/scratch/avaidya/data/afhq/val",
        'seed': 69,
        'labels': ["cat", "dog"],  
    }
}