"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint
from argparse import Namespace
import torch
import torch.nn as nn 

sys.path.append(".")
sys.path.append("..")

sys.path.append("./")
sys.path.append("../")


from training.coach import Coach


def main():

    args = Namespace(device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
                    train_dir = "../data/afhq/train",
                    val_dir = "../data/afhq/val",
                    save_path = "./checkpoints",
                    exp_dir = "./args",
                    log_dir = "./",
                    seed = 7,
                    labels = ["cat", "dog"],
                    batch_size = 64,
                    test_batch_size = 64,
                    epochs = 50,
                    num_workers = 0,
                    class_names = {0:"cat", 1:"dog"} ,
                    lr = 0.0001,
                    lr_d = 0.0004,
                    momentum = 0.9,
                    criterion = nn.CrossEntropyLoss(),
                    optim_name = "ranger",
                    scheduler = "STEP",
                    scheduler_step_size = 7,
                    scheduler_gamma = 0.1,
                    exp_name = "stylespace1",
                    wandb_config = {"learning_rate": 0.0001, "epochs": 2, "batch_size": 64},
                    use_wandb = True,
                    wandb_interval = 50,
                    output_size = 512,
                    encoder_type = "gradual",
                    n_styles = 0,
                    lambdas = {"adv_d":1,"adv_g":1, "reg":1, "rec_x":1, "rec_w":1, "lpips":1, "clf":1, "r1" : 10},
                    train_decoder = True, # whether to train decoder,
                    dataset_type = "afhq",
                    max_steps = 50000, # max number of training steps,
                    save_interval = 5000, # checkpoint saving interval,
                    start_from_latent_avg = False, #Whether to add average latent vector to generate codes from encoder
                    learn_in_w = True, # Whether to learn in w space instead of w+,
                    img_size = 512, #image sizes for the model
                    channel_multiplier = 2, # channel multiplier factor for the model. config-f = 2, else = 1,
                    val_interval = 1000, #validation interval,
                    d_reg_every = 16, # interval of the applying r1 regularization,
                    g_reg_every = 4, # interval of the applying path length regularization
    )
	
    if os.path.exists(args.exp_dir):
        raise Exception('Oops... {} already exists'.format(args.exp_dir))
    os.makedirs(args.exp_dir)

    args_dict = vars(args)
    pprint.pprint(args_dict)
    with open(os.path.join(args.exp_dir, 'opt.json'), 'w') as f:
        json.dump(args_dict, f, indent=4, sort_keys=True)

    coach = Coach(args)
    coach.train()


if __name__ == '__main__':
	main()