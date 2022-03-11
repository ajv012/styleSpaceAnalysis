import os
import matplotlib
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import pdb
matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim

from utils import common, train_utils
from criteria.d_r1_loss import d_r1_loss
from criteria.adv_loss import adv_loss
from criteria.clf_loss import clf_loss
from criteria.path_reg_loss import path_reg_loss
from criteria.lpips.lpips import LPIPS
from configs import data_configs
from datasets.afhq_dataset import afhq_dataset

from models.encoders import encoders
from models.stylegan2.model import Generator
from models.discriminator.model import Discriminator
from models.classifier import Classifier

from utils.non_leaky import augment, AdaptiveAugment
from utils.wandb_utils import WBLogger
from training.ranger import Ranger


class Coach:
    def __init__(self, args):
        self.args = args

        self.global_step = 0

        # accumaltion decay factor
        self.accum = 0.5 ** (32 / (10 * 1000))
        self.r_t_stat = 0

        # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES (use distributed module)
        self.device = self.args.device

        if self.args.use_wandb:
            self.wb_logger = WBLogger(self.args)

        # Initialize all the networks
        models_init = self.init_models()
        print(models_init)
        self.ada_aug_p = self.args.augment_p if self.args.augment_p > 0 else 0.0

        if self.args.augment and self.args.augment_p == 0:
            self.ada_augment = AdaptiveAugment(
                ada_aug_target = self.args.ada_target, 
                ada_aug_len = self.args.ada_length, 
                update_every = 8, 
                device = self.device
            )

        # Initialize loss
        losses_init = self.init_losses(self.args)
        print(losses_init)

        # Initialize optimizer
        self.optimizer_e, self.optimizer_g, self.optimizer_d = self.configure_optimizers()
        print("initiailized optimizers for encoder, generator, decoder")

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        print("created train and test datasets. Train len = {}, Test len = {}".format(len(self.train_dataset),
                                                                                      len(self.test_dataset)))
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=int(self.args.num_workers),
            drop_last=True
        )
        print("created train dataloader")
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=int(self.args.num_workers),
            drop_last=True
        )

        # Initialize logger
        log_dir = os.path.join(self.args.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)
        print("made log dir {} and created Tensorboard summary writer".format(log_dir))

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(self.args.exp_dir, 'checkpoints', 'cat_dog_styleEx')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print("made checkpoints dir {}".format(self.checkpoint_dir))

        self.best_val_loss = None
        if self.args.save_interval is None:
            self.args.save_interval = self.args.max_steps

    def init_models(self):
        # calc num styles
        self.args.n_styles = int(math.log(self.args.output_size, 2)) * 2 - 2

        # initialize encoder
        self.encoder = encoders.GradualStyleEncoder(image_size=self.args.img_size, num_layers=self.args.num_enc_layers,
                                                    mode=self.args.mode_enc, opts=self.args).to(self.device)

        # initialize decoder
        # self.decoder = Generator(self.args.output_size, style_dim=self.args.latent_dim, c_dim=2,
        #                          n_mlp=self.args.n_mlp, channel_multiplier=self.args.channel_multiplier).to(self.device)
        self.decoder = Generator(self.args.output_size, style_dim=self.args.latent_dim, c_dim=0,
                                 n_mlp=self.args.n_mlp, channel_multiplier=self.args.channel_multiplier).to(self.device)

        # initialize decoder to keep ema
        # self.decoder_ema = Generator(self.args.output_size, style_dim=self.args.latent_dim, c_dim=2,
        #                          n_mlp=self.args.n_mlp, channel_multiplier=self.args.channel_multiplier).to(self.device)
        self.decoder_ema = Generator(self.args.output_size, style_dim=self.args.latent_dim, c_dim=0,
                                 n_mlp=self.args.n_mlp, channel_multiplier=self.args.channel_multiplier).to(self.device)
        self.decoder_ema.eval()
        self.accumulate(model1 = self.decoder_ema, model2 = self.decoder, decay = 0)
        
        # initialize discriminator
        self.discriminator = Discriminator(self.args.img_size, self.args.channel_multiplier).to(self.device)

        # initialize clf
        self.classifier = Classifier(self.args).to(self.device)
        state_dict = torch.load(self.args.path_to_weights, map_location=self.device)
        self.classifier.load_state_dict(state_dict["model_state_dict"])
        self.classifier.eval()

        return "models initialized"

    def init_losses(self, args):
        # adv loss
        # if args.lambdas["adv_d"] > 0:
        # adv loss requires generative part as well
        self.adv_loss = adv_loss().to(self.device)
        print("initiailized adv loss")
        # path regularization for generator
        # if args.lambdas["reg"] > 0:
        self.reg_loss = path_reg_loss().to(self.device)
        print("initiailized path length regularization loss")
        # rec_x
        # if args.lambdas["rec_x"] > 0:
        self.rec_x_loss = nn.L1Loss().to(self.device)
        print("initiailized rec_x loss")
        # lpips
        # if args.lambdas["lpips"] > 0:
        self.lpips_loss = LPIPS(net_type='alex').to(self.device)
        print("initiailized lpips loss")
        # rec_w
        # if args.lambdas["rec_w"] > 0:
        self.rec_w_loss = nn.L1Loss().to(self.device)
        print("initiailized rec_w loss")
        # clf
        # if args.lambdas["clf"] > 0:
        self.clf_loss = clf_loss(self.classifier, self.args).to(self.device)
        print("initialized clf loss")
        # discriminator regularization loss
        # if args.lambdas["r1"] > 0:
        self.d_r1_loss = d_r1_loss(self.args).to(self.device)
        print("initiailized r1 loss")
        return "all losses created"

    def configure_optimizers(self):
        # encoder + decoder optim
        params_g = self.decoder.parameters()
        params_e = self.encoder.parameters()
        params_d = self.discriminator.parameters()

        g_optimizer = optim.Adam(params_g, lr=self.args.lr_g)
        e_optimizer = optim.Adam(params_e, lr=self.args.lr_g)
        d_optimizer = optim.Adam(params_d, lr=self.args.lr_d)

        return e_optimizer, g_optimizer, d_optimizer

    def configure_datasets(self):
        print(f'Loading dataset for {self.args.dataset_type}')
        dataset_args = data_configs.DATASETS[self.args.dataset_type]
        transforms_dict = dataset_args['transforms'](self.args).get_transforms()
        train_dataset = afhq_dataset(
            dataset_args["train_dir"],
            dataset_args["seed"],
            dataset_args["labels"],
            transforms_dict["transform_train"]
        )

        val_dataset = afhq_dataset(
            dataset_args["val_dir"],
            dataset_args["seed"],
            dataset_args["labels"],
            transforms_dict["transform_val"]
        )

        if self.args.use_wandb:
            self.wb_logger.log_dataset_wandb(train_dataset, dataset_name="Train", device=self.device)
            self.wb_logger.log_dataset_wandb(val_dataset, dataset_name="Val", device=self.device)

        return train_dataset, val_dataset

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train(self):

        self.set_train_status(train=True)
        mean_path_length = 0

        while self.global_step < self.args.max_steps:

            for batch_idx, batch in enumerate(self.train_dataloader):

                # get all data
                x_all, y_all = batch["inputs"], batch["labels"]

                ########### cGAN (works with noise and x_1) ###########
                x_1, y_1 = x_all[0], y_all[0]
                # x_2, y_2 = x_all[1], y_all[1]

                x_1, y_1 = x_1.to(self.device).float(), y_1.to(self.device).float()
                # x_2, y_2 = x_2.to(self.device).float(), y_2.to(self.device).float()

                # get conditioning output, i.e. clf out on image
                # with torch.no_grad():
                #     conditioning_1 = self.classifier(x_1)
                #     conditioning_2 = self.classifier(x_2)

                #################### Discriminator update ##########################################
                # self.requires_grad(self.encoder, False)
                self.requires_grad(self.decoder, False)
                self.requires_grad(self.discriminator, True)

                # make noise (follow repo)
                noise = torch.randn(self.args.batch_size, self.args.latent_dim, device=self.device)

                # get output of generator
                # y_1_hat, latent_1 = self.decoder(
                #     styles=[noise],
                #     conditioning=conditioning_1,
                #     use_style_encoder=True,
                #     return_latents=True
                # )

                y_1_hat, latent_1 = self.decoder(
                    styles=[noise],
                    conditioning=None,
                    use_style_encoder=True,
                    return_latents=True
                )

                # discriminator
                d_regularize = self.global_step % self.args.d_reg_every == 0
                if d_regularize:
                    which_loss = ["adv_d", "r1"]
                    x_1.requires_grad = True
                else:
                    which_loss = ["adv_d"]

                # augment images before sending to discriminator 
                if self.args.augment:
                    real_img_aug, _ = augment(x_1, self.ada_aug_p)
                    fake_img, _ = augment(y_1_hat, self.ada_aug_p)
                else:
                    real_img_aug = x_1
                    fake_img = y_1_hat

                # use x1 to get discriminator outputs
                real_pred_1 = self.discriminator(real_img_aug)
                fake_pred_1 = self.discriminator(fake_img)

                discriminator_loss, discriminator_loss_dict, _ = self.calc_loss(
                    x = x_1,
                    x_aug=real_img_aug,
                    fake_pred=fake_pred_1,
                    real_pred=real_pred_1,
                    loss_type=which_loss
                )

                # discriminator
                self.discriminator.zero_grad()
                discriminator_loss.backward()
                self.optimizer_d.step()

                # update augments
                if self.args.augment and self.args.augment_p == 0:
                    self.ada_aug_p = self.ada_augment.tune(real_pred_1)
                    self.r_t_stat = self.ada_augment.r_t_stat

                ##################################################################################
                #################### Generator update ################################
                # self.requires_grad(self.encoder, False)
                self.requires_grad(self.decoder, True)
                self.requires_grad(self.discriminator, False)

                # make noise (follow repo)
                noise = torch.randn(self.args.batch_size, self.args.latent_dim, device=self.device)

                # get output of generator
                # y_1_hat, latent_1 = self.decoder(
                #     styles=[noise],
                #     conditioning=conditioning_1,
                #     use_style_encoder=True,
                #     return_latents=True
                # )

                y_1_hat, latent_1 = self.decoder(
                    styles=[noise],
                    conditioning=None,
                    use_style_encoder=True,
                    return_latents=True
                )

                if self.args.augment:
                    fake_img, _ = augment(y_1_hat, self.ada_aug_p)
                else:
                    fake_img = y_1_hat

                # use x1 to get discriminator outputs
                real_pred_1 = self.discriminator(real_img_aug)
                fake_pred_1 = self.discriminator(fake_img)

                # generator (adversarial losses)
                g_regularize = self.global_step % self.args.g_reg_every == 0
                if g_regularize:
                    which_loss = ["adv_g", "reg"]
                else:
                    which_loss = ["adv_g"]
                generator_loss, generator_loss_dict, mean_path_length = self.calc_loss(
                    x = x_1,
                    y_hat=y_1_hat,
                    y_hat_aug=y_1_hat,
                    latent=latent_1,
                    fake_pred=fake_pred_1,
                    real_pred=real_pred_1,
                    mean_path_length=mean_path_length,
                    loss_type=which_loss
                )

                ### encoder part
                # get encodings
                # encoder_rep_x_2 = self.encoder(x_2)

                # # get output of generator
                # y_2_hat, latent_2 = self.decoder(
                #     styles=[encoder_rep_x_2],
                #     conditioning=conditioning_2,
                #     use_style_encoder=False,
                #     return_latents=True
                # )

                # # get encoding of y_2_hat for loss purposes
                # w_fake_2 = self.encoder(y_2_hat)

                ########### calculate losses ###########
                # reconstruction losses
                # which_loss = ["rec_x", "rec_w"]
                # recon_loss, recon_loss_dict, _ = self.calc_loss(
                #     x=x_2,
                #     y_hat=y_2_hat,
                #     w_fake=w_fake_2,
                #     w_real=encoder_rep_x_2,
                #     loss_type=which_loss
                # )

                # which_loss = ["rec_x"]
                # recon_loss, recon_loss_dict, _ = self.calc_loss(
                #     x=x_2,
                #     y_hat=y_2_hat,
                #     loss_type=which_loss
                # )
                

                # which_loss = ["lpips"]
                # perceptual_loss, perceptual_loss_dict, _ = self.calc_loss(
                #     x=x_2,
                #     y_hat=y_2_hat,
                #     loss_type=which_loss
                # )

                which_loss = ["rec_x"]
                recon_loss, recon_loss_dict, _ = self.calc_loss(
                    x=x_1,
                    y_hat=y_1_hat,
                    loss_type=which_loss
                )
                

                which_loss = ["lpips"]
                perceptual_loss, perceptual_loss_dict, _ = self.calc_loss(
                    x=x_1,
                    y_hat=y_1_hat,
                    loss_type=which_loss
                )

                # which_loss = ["clf"]
                # cycle_loss, cycle_loss_dict, _ = self.calc_loss(
                #     x=x_2,
                #     y_hat=y_2_hat,
                #     loss_type=which_loss
                # )
                cycle_loss = 0
                cycle_loss_dict = {}
                generator_loss += (recon_loss + perceptual_loss + cycle_loss)

                self.decoder.zero_grad()
                generator_loss.backward()
                self.optimizer_g.step()

                # update the ema model
                self.accumulate(model1 = self.decoder_ema, model2 = self.decoder, decay = self.accum)

                ##################################################################################
                #################### Encoder update ####################################
                # self.requires_grad(self.encoder, True)
                # self.requires_grad(self.decoder, False)
                # self.requires_grad(self.discriminator, False)

                # ########### autoencoder (works with encoder and x_2) ###########
                # # get encodings
                # encoder_rep_x_2 = self.encoder(x_2)

                # # get output of generator
                # y_2_hat, latent_2 = self.decoder(
                #     styles=[encoder_rep_x_2],
                #     conditioning=conditioning_2,
                #     use_style_encoder=False,
                #     return_latents=True
                # )

                # # get encoding of y_2_hat for loss purposes
                # w_fake_2 = self.encoder(y_2_hat)

                # ########### calculate losses ###########
                # # reconstruction losses
                # which_loss = ["rec_x", "rec_w"]
                # recon_loss, recon_loss_dict, _ = self.calc_loss(
                #     x=x_2,
                #     y_hat=y_2_hat,
                #     w_fake=w_fake_2,
                #     w_real=encoder_rep_x_2,
                #     loss_type=which_loss
                # )

                # which_loss = ["lpips"]
                # perceptual_loss, perceptual_loss_dict, _ = self.calc_loss(
                #     x=x_2,
                #     y_hat=y_2_hat,
                #     loss_type=which_loss
                # )

                # which_loss = ["clf"]
                # cycle_loss, cycle_loss_dict, _ = self.calc_loss(
                #     x=x_2,
                #     y_hat=y_2_hat,
                #     loss_type=which_loss
                # )
                # # print("going into trace 2")
                # # pdb.set_trace()
                # # combine losses to get generator and encoder losses
                # encoder_loss = perceptual_loss + cycle_loss + recon_loss

                # ########### backpropagate ###########
                # # encoder
                # self.encoder.zero_grad()
                # encoder_loss.backward()
                # self.optimizer_e.step()

                # all losses
                loss_dict = dict(discriminator_loss_dict.items() | generator_loss_dict.items() | recon_loss_dict.items() | perceptual_loss_dict.items() | cycle_loss_dict.items())
                loss_dict["loss"] = sum([loss_dict[key] for key in loss_dict if key != "loss"])
                loss_dict["rt"] = self.r_t_stat

                # Logging related
                if self.global_step % self.args.wandb_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Log images of first batch to wandb
                if self.args.use_wandb and batch_idx == 0:
                    self.wb_logger.log_images_to_wandb(x_1, y_1, y_1_hat, prefix="train", step=self.global_step,
                                                       opts=self.args)

                # # Validation related
                val_loss_dict = None
                if self.global_step % self.args.val_interval == 0 or self.global_step == self.args.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.args.save_interval == 0 or self.global_step == self.args.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.args.max_steps:
                    print('OMG, Holy Moly, Thank God, finished training!')
                    break

                self.global_step += 1

    def accumulate(self, model1, model2, decay=0.999):
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
    
    def set_train_status(self, train=True):
        if train:
            self.encoder.train()
            self.decoder.train()
            self.discriminator.train()
            print("set all model status to train")
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.discriminator.eval()
            print("set all model status to eval")

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
                if self.args.use_wandb:
                    self.wb_logger.log_best_model()
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def calc_loss(self, x=None, x_aug=None, y_hat=None, y_hat_aug=None, latent=None, fake_pred=None, real_pred=None, w_fake=None, w_real=None,
                  mean_path_length=None, loss_type=None):
        r"""
        loss_type is a list
        """
        loss_dict = {}
        loss = torch.Tensor([0.0]).to(self.device)
        types = ["adv_d", "adv_g", "reg", "rec_x", "lpips", "rec_w", "clf", "r1"]

        for curr_loss_name in loss_type:
            assert curr_loss_name in types, "Invalid loss name"
            # adversarial losses
            # adv
            if curr_loss_name == "adv_d":
                loss_adv = self.adv_loss(real_pred, fake_pred, disc=True)
                loss_dict["adv_loss_d"] = self.args.lambdas["adv_d"] * loss_adv
                loss += loss_dict["adv_loss_d"]
            if curr_loss_name == "r1":
                loss_r1 = self.d_r1_loss(real_pred, x_aug)
                loss_dict["r1_loss"] = self.args.lambdas["r1"] * loss_r1
                loss += loss_dict["r1_loss"]
            if curr_loss_name == "adv_g":
                loss_adv = self.adv_loss(real_pred, fake_pred, disc=False)
                loss_dict["adv_loss_g"] = self.args.lambdas["adv_g"] * loss_adv
                loss += loss_dict["adv_loss_g"]
            # path regularization
            if curr_loss_name == "reg":
                loss_reg, mean_path_length, path_lengths = self.reg_loss(y_hat_aug, latent, mean_path_length)
                weighted_reg_loss = self.args.lambdas["reg"] * self.args.g_reg_every * loss_reg
                weighted_reg_loss += 0 * y_hat_aug[0, 0, 0, 0]
                loss_dict["reg"] = weighted_reg_loss
                loss += loss_dict["reg"]

            # reconstruction losses
            # rec_x
            if curr_loss_name == "rec_x":
                loss_rec_x = self.rec_x_loss(x, y_hat)
                loss_dict["rec_x"] = self.args.lambdas["rec_x"] * loss_rec_x
                loss += loss_dict["rec_x"]
            # lpips
            if curr_loss_name == "lpips":
                loss_lpips = self.lpips_loss(x, y_hat)
                loss_dict["lpips"] = self.args.lambdas["lpips"] * loss_lpips
                loss += loss_dict["lpips"]
            # rec_w
            if curr_loss_name == "rec_w":
                loss_rec_w = self.rec_w_loss(w_fake, w_real)
                loss_dict["rec_w"] = self.args.lambdas["rec_w"] * loss_rec_w
                loss += loss_dict["rec_w"]
            # clf
            if curr_loss_name == "clf":
                loss_clf = self.clf_loss(x, y_hat)
                loss_dict["clf"] = self.args.lambdas["clf"] * loss_clf
                loss += loss_dict["clf"]

        loss_dict['loss'] = float(loss)
        return loss, loss_dict, mean_path_length

    def validate(self):
        self.set_train_status(train=False)
        agg_loss_dict = []
        for batch_idx, batch in tqdm(enumerate(self.test_dataloader), total = len(self.test_dataloader.dataset), desc = "validation"):
            
            x_all, y_all = batch["inputs"], batch["labels"]

            with torch.no_grad():
                # during validation only use the autoencoder branch
                x_1, y_1 = x_all[0], y_all[0]
                x_1, y_1 = x_1.to(self.device).float(), y_1.to(self.device).float()

                # get conditioning
                # conditioning_1 = self.classifier(x_1)

                # make noise only when doing just stylegan
                noise = torch.randn(self.args.test_batch_size, self.args.latent_dim, device=self.device)

                # get encodings
                # encoder_rep_x_1 = self.encoder(x_1)

                # get output of generator
                # y_1_hat, latent_2 = self.decoder_ema(
                #     styles=[encoder_rep_x_1],
                #     conditioning=conditioning_1,
                #     use_style_encoder=False,
                #     return_latents=True
                # )

                y_1_hat, latent_1 = self.decoder_ema(
                    styles=[noise],
                    conditioning=None,
                    use_style_encoder=True,
                    return_latents=True
                )

                # w_fake_1 = self.encoder(y_1_hat)

                # calculate losses
                # which_loss = ["rec_x", "rec_w"]
                # recon_loss, recon_loss_dict, _ = self.calc_loss(
                #     x=x_1,
                #     y_hat=y_1_hat,
                #     w_fake=w_fake_1,
                #     w_real=encoder_rep_x_1,
                #     loss_type=which_loss
                # )

                which_loss = ["rec_x"]
                recon_loss, recon_loss_dict, _ = self.calc_loss(
                    x=x_1,
                    y_hat=y_1_hat,
                    loss_type=which_loss
                )

                which_loss = ["lpips"]
                perceptual_loss, perceptual_loss_dict, _ = self.calc_loss(
                    x=x_1,
                    y_hat=y_1_hat,
                    loss_type=which_loss
                )

                # which_loss = ["clf"]
                # cycle_loss, cycle_loss_dict, _ = self.calc_loss(
                #     x=x_1,
                #     y_hat=y_1_hat,
                #     loss_type=which_loss
                # )
                cycle_loss = 0
                cycle_loss_dict = {}

                # combine losses
                loss_dict = dict(recon_loss_dict.items() | perceptual_loss_dict.items() | cycle_loss_dict.items())
                loss_dict["loss"] = sum([loss_dict[key] for key in loss_dict if key != "loss"])

            agg_loss_dict.append(loss_dict)

            # Logging related
            # saving all generated images with the titles on disk
            self.parse_and_log_images(
                x_1, y_1, y_1_hat,
                title='images/test/dogs-cats',
                subscript='{:04d}'.format(batch_idx),
                display_count = 1
            )

            # Log images of first batch to wandb (number of images on wandb will be same as batch size)
            if self.args.use_wandb and batch_idx == 0:
                self.wb_logger.log_images_to_wandb(x_1, y_1, y_1_hat, prefix="test", step=self.global_step,
                                                   opts=self.args)

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                print('Performing only sanity check for first batch')
                self.set_train_status(train=True)
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)

        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.set_train_status(train=True)
        return loss_dict

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
        if self.args.use_wandb:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def parse_and_log_images(self, x, y, y_hat, title, subscript=None, display_count=1):
        im_data = []
        # TODO: pass as a single batch
        for i in range(display_count):
            # for current x, get clf decision and top class probability
            curr_x = x[i].unsqueeze(0)
            out_x = self.classifier(curr_x)
            values_x, preds_x = torch.max(out_x, 1)

            # for current y_hat, get clf decision and top class probability
            curr_y_hat =  y_hat[i].unsqueeze(0)
            out_y_hat = self.classifier(curr_y_hat)
            values_y_hat, preds_y_hat = torch.max(F.softmax(out_y_hat, dim = 1), 1)

            # define title info
            title_info = {
                "true_label": y[i],
                "pred_label_x": preds_x,
                "pred_label_y_hat": preds_y_hat,
                "top_score_x": values_x,
                "top_score_y_hat": values_y_hat
            }

            cur_im_data = {
                'input': common.tensor2im(x[i]),
                # selecting images of size [num_channels, H, W] and converting them to PIL images
                'output': common.tensor2im(y_hat[i]),
                "title_info": title_info
            }
            im_data.append(cur_im_data)  # appending dictionaries
        self.log_images(title, im_data=im_data, subscript=subscript)  # im_data is a list of dictionaries

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_outputs(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'generator_state_dict': self.decoder_ema.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            "optim_g": self.optimizer_g,
            "optim_e": self.optimizer_e,
            "optim_d": self.optimizer_d,
            'opts': vars(self.args)
        }
        return save_dict