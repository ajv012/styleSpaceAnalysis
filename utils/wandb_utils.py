import datetime
import os
import numpy as np
import wandb

from utils import common

class WBLogger:

    def __init__(self, args):
        wandb_run_name = args.exp_name
        wandb.init(project="style_space_run", config=vars(args.wandb_config), name=wandb_run_name, entity="stylespace")

    @staticmethod
    def log_best_model():
        wandb.run.summary["best-model-save-time"] = datetime.datetime.now()

    @staticmethod
    def log_dataset_wandb(dataset, dataset_name, device, n_images=16):
        idxs = np.random.choice(a=range(len(dataset)), size=n_images, replace=False)
        data = [wandb.Image(common.tensor2im(dataset[idx]["inputs"])) for idx in idxs]
        wandb.log({f"{dataset_name} Data Samples": data})

    @staticmethod
    # method should log all relevant metrics for a run
    def log(prefix, metrics_dict, global_step):
        log_dict = {f'{prefix}_{key}': value for key, value in metrics_dict.items()} 
        log_dict["global_step"] = global_step
        wandb.log(log_dict)

    @staticmethod
    def log_images_to_wandb(x, y, y_hat, prefix, step, opts):
        im_data = []
        column_names = ["Source", "Target", "Output"]
        for i in range(len(x)):
            cur_im_data = [
                wandb.Image(common.tensor2im(x[i])),
                wandb.Image(common.tensor2im(y[i])),
                wandb.Image(common.tensor2im(y_hat[i])),
            ]
            im_data.append(cur_im_data)
        outputs_table = wandb.Table(data=im_data, columns=column_names)
        wandb.log({f"{prefix.title()} Step {step} Output Samples": outputs_table})