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

# sys.path.append(".")
# sys.path.append("..")


from training.coach_memory_efficient import Coach


def main():
    print("in main")
    args = Namespace(device="cuda:0",
                    train_dir = "/data/vision/polina/scratch/avaidya/data/afhq/train",
                    val_dir = "/data/vision/polina/scratch/avaidya/data/afhq/val",
                    # exp_dir = "/data/vision/torralba/scratch/swamiviv/stylex_afhq_cat_dog",
                    exp_dir = "/data/vision/polina/scratch/avaidya/styleSpaceAnalysis/",
                    seed = 7,
                    labels = ["cat", "dog"],
                    batch_size = 2,
                    test_batch_size = 2,
                    epochs = 50,
                    num_workers = 1,
                    class_names = {0:"cat", 1:"dog"} ,
                    lr_g = 0.001,
                    lr_d = 0.001,
                    momentum = 0.9,
                    optim_name = "ranger",
                    scheduler = "STEP",
                    scheduler_step_size = 7,
                    scheduler_gamma = 0.1,
                    exp_name = "stylespace_analysis_catdog",
                    wandb_config={"learning_rate": 0.0001, "epochs": 2, "batch_size": 64},
                    use_wandb=True,
                    wandb_interval=50,
                    output_size = 128, #Stylegan decoder output size
                    encoder_type = "gradual",
                    n_styles = 0,
                    lambdas = {"adv_d":1.,"adv_g":1., "reg":1., "rec_x":1., "rec_w":1., "lpips":1., "clf":1., "r1" : 5},
                    train_decoder = True, # whether to train decoder,
                    dataset_type = "afhq",
                    max_steps = 50000, # max number of training steps,
                    save_interval = 100, # checkpoint saving interval,
                    start_from_latent_avg = False, #Whether to add average latent vector to generate codes from encoder
                    learn_in_w = True, # Whether to learn in w space instead of w+,
                    img_size = 128, #image sizes for the model
                    channel_multiplier = 1, # channel multiplier factor for the model. config-f = 2, else = 1,
                    val_interval = 1000, #validation interval,
                    d_reg_every = 16, # interval of the applying r1 regularization,
                    g_reg_every = 4, # interval of the applying path length regularization
                     # TODO: crashes when this is changed - need debugging
                    latent_dim = 256, # latent dim of stylegan W network,
                    num_enc_layers = 50, # number of layers in gradual style encoder,
                    mode_enc = "ir_se", # mode for gradual style encoder 
                    input_nc = 3, # number of input channels in img
                    n_mlp = 8, # number of mlp in stylegan,
                    path_to_weights = "/data/vision/polina/scratch/avaidya/styleSpaceAnalysis/checkpoints/cat_dog_weights/checkpoint_2.pt",
                    log_image_interval = 100,
    )
    print("defined args") 

    os.makedirs(args.exp_dir, exist_ok=True)
    print("Made experiment directory")
    
    args_dict = vars(args)
    pprint.pprint(args_dict)
    with open(os.path.join(args.exp_dir, 'opt.json'), 'w') as f:
        json.dump(args_dict, f, indent=4, sort_keys=True)
    print("Dumped the args in a json")

    coach = Coach(args)
    print("Created coach, about to start training")
    
    coach.train()


if __name__ == '__main__':
    main()

