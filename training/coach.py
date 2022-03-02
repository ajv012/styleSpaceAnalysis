import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim

from utils import common, train_utils
from criteria import adv_loss, clf_loss, path_reg_loss, d_r1_loss
from criteria.lpips.lpips import LPIPS
from configs import data_configs
from datasets.afhq_dataset import afhq_dataset

from models.encoders import encoders
from models.stylegan2.model import Generator
from models.discriminator.model import Discriminator
from models.classifier import Classifier

from utils.wandb_utils import WBLogger

from training.ranger import Ranger

class Coach:
	def __init__(self, args):
		self.args = args

		self.global_step = 0

		# TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES (use distributed module)
		self.device = self.args.device  

		if self.args.use_wandb:
			self.wb_logger = WBLogger(self.args)
		
		# Initialize all the networks
		models_init = self.init_models()
		print(models_init)

		# Estimate latent_avg via dense sampling if latent_avg is not available
		# if self.net.latent_avg is None:
		# 	self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()

		# Initialize loss
		# adv loss
		if self.args.lambdas["adv"] > 0:
			# adv loss requires generative part as well 
			self.adv_loss = adv_loss().to(self.device) 
		# path regularization for generator
		if self.args.lambdas["reg"] > 0:
			self.reg_loss = path_reg_loss()
		# rec_x
		if self.args.lambdas["rec_x"] > 0:
			self.rec_x_loss = nn.L1Loss().to(self.device)
		# lpips
		if self.args.lambdas["lpips"] > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device)
		# rec_w
		if self.args.lambdas["rec_w"] > 0:
			self.rec_w_loss = nn.L1Loss().to(self.device)
		# clf
		if self.args.lambdas["clf"] > 0:
			self.clf_loss = clf_loss(self.args)
		# discriminator regularization loss
		if self.args.lambdas["r1"] > 10:
			self.d_r1_loss = d_r1_loss(self.args)

		# Initialize optimizer
		self.optimizer_e, self.optimizer_g, self.optimizer_d = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
											batch_size=self.args.batch_size,
											shuffle=True,
											num_workers=int(self.opts.workers),
											drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
											batch_size=self.args.test_batch_size,
											shuffle=False,
											num_workers=int(self.opts.test_workers),
											drop_last=True)

		# Initialize logger
		log_dir = os.path.join(self.args.log_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints', 'cat_dog_styleEx')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.args.save_interval is None:
			self.args.save_interval = self.args.max_steps

	def init_models(self):
		# calc num styles 
		self.args.n_styles = int(math.log(self.args.output_size, 2)) * 2 - 2

		# initialize encoder
		self.encoder = encoders.GradualStyleEncoder(num_layers=self.args.num_enc_layers, mode=self.args.mode_enc, self.args).to(self.device) 

		# initialize decoder
		self.decoder = Generator(self.args.output_size, style_dim = self.args.latent_dim, n_mlp = self.args.n_mlp).to(self.device)

		# initialize discriminator
		self.discriminator = Discriminator(self.args.img_size, self.channel_multiplier).to(self.device)

		# initialize clf
		self.classifier = Classifier(self.args).to(self.device)

		return "models initialized"
	
	def configure_optimizers(self):
		# encoder + decoder optim 
		params_g = self.decoder.parameters()
		params_e = self.encoder.parameters()
		params_d = self.discriminator.parameters()

		g_optimizer = optim.Adam(params_g, lr=self.args.lr_g)
		e_optimizer = Ranger(params_e, lr=self.args.lr_g)
		d_optimizer = optim.Adam(params_d, lr=self.args.lr_d)

		return e_optimizer, g_optimizer, d_optimizer

	def configure_datasets(self):
		print(f'Loading dataset for {self.args.dataset_type}')
		dataset_args = data_configs.DATASETS[self.args.dataset_type]
		transforms_dict = dataset_args['transforms'](self.args).get_transforms()
		train_dataset = afhq_dataset(dataset_args["train_dir"],
									 dataset_args["seed"], 
									 dataset_args["labels"], 
									 transforms_dict["transform_train"])
									
		val_dataset = afhq_dataset(dataset_args["val_dir"],
									 dataset_args["seed"], 
									 dataset_args["labels"], 
									 transforms_dict["transform_val"])
		if self.args.use_wandb:
			self.wb_logger.log_dataset_wandb(train_dataset, dataset_name="Train")
			self.wb_logger.log_dataset_wandb(val_dataset, dataset_name="Val")
		print(f"Number of training samples: {len(train_dataset)}")
		print(f"Number of test samples: {len(val_dataset)}")
		return train_dataset, val_dataset

	def requires_grad(self, model, flag=True):
		for p in model.parameters():
			p.requires_grad = flag

	def train(self):

		self.set_train_status(train = True)
		mean_path_length = 0 

		while self.global_step < self.args.max_steps:

			for batch_idx, batch in enumerate(self.train_dataloader):
				
				# get all data
				x_all, y_all = batch["inputs"], batch["labels"]
				
				########### cGAN (works with noise and x_1) ###########
				x_1, y_1 = x_all[0], y_all[0]
				x_1, y_1 = x_1.to(self.device).float(), y_1.to(self.device).float()

				# make noise (follow repo)
				noise = torch.randn(self.args.batch_size, self.args.latent_dim, device=self.device)

				# get conditioning output, i.e. clf out on image
				conditioning_1 = self.classifier(x_1)

				# get output of generator
				y_1_hat, latent_1 = self.generator(style = noise, 
												   conditioning = conditioning_1, 
												   use_style_encoder = True, 
												   return_latents = True)

				# use x1 to get discriminator outputs
				real_pred_1 = self.discriminator(x_1)
				fake_pred_1 = self.discriminator(y_1_hat)

				# use everything associated with x1 for adversarial losses

				########### autoencoder (works with encoder and x_2) ###########
				x_2, y_2 = x_all[1], y_all[1]
				x_2, y_2 = x_2.to(self.device).float(), y_2.to(self.device).float()

				# get conditioning
				conditioning_2 = self.classifier(x_2)	

				# get encodings
				E_2 = self.encoder(x_2)

				# get output of generator	
				y_2_hat, latent_2 = self.generator(style = E_2, 
												   conditioning = conditioning_2, 
												   use_style_encoder = False, 
												   return_latents = True)
				
				# get encoding of y_2_hat for loss purposes
				self.encoder.eval()
				w_fake_2 = self.encoder(y_2_hat)
				self.encoder.train()
				
				########### calculate losses ###########
				
				# discriminator 
				d_regularize = self.global_step % self.args.d_reg_every == 0
				if d_regularize:
					which_loss = ["adv_d", "r1"]
				else:
					which_loss = ["adv_d"]

				discriminator_loss, discriminator_loss_dict, _ = self.calc_loss(x_1, 
																				fake_pred = fake_pred_1, 
																				real_pred = real_pred_1, 
																				loss_type=which_loss)
				
				# generator (adversarial losses)
				g_regularize = self.global_step % self.args.g_reg_every == 0
				if g_regularize:
					which_loss = ["adv_g", "reg"]
				else:
					which_loss = ["adv_g"]
				generator_loss, generator_loss_dict, mean_path_length = self.calc_loss(x_1, 
																					   y_hat = y_1_hat,
																					   latent = latent_1,
																					   fake_pred = fake_pred_1,
																					   real_pred = real_pred_1,
												 									   mean_path_length,
																					   loss_type=which_loss)

				# reconstruction losses
				which_loss = ["rec_x", "rec_w"]
				recon_loss, recon_loss_dict, _ = self.calc_loss(x = x_2,
											y_hat = y_2_hat,
											w_fake = w_fake_2,
											w_real = E_2,
											loss_type = which_loss)
				
				which_loss = ["lpips"]
				perceptual_loss, perceptual_loss_dict, _ = self.calc_loss(x = x_2, 
												 y_hat = y_2_hat,
											     loss_type = which_loss)

				which_loss = ["clf"]
				cycle_loss, cycle_loss_dict, _ = self.calc_loss(x = x_2,
											y_hat = y_2_hat,
											loss_type = which_loss)


				# combine losses to get generator and encoder losses
				generator_loss = generator_loss + recon_loss + perceptual_loss + cycle_loss
				encoder_loss = recon_loss + perceptual_loss + cycle_loss

				########### backpropogate ###########
				# discriminator
				self.requires_grad(self.decoder, False)
				self.requires_grad(self.discriminator, True)

				self.optimizer_d.zero_grad()
				discriminator_loss.backward() 
				self.optimizer_d.step()	

				# generator
				self.requires_grad(self.decoder, True)
				self.requires_grad(self.discriminator, False)

				self.optimizer_g.zero_grad()
				generator_loss.backward()
				self.optimizer_g.step()

				# encoder		
				self.optimizer_e.zero_grad()
				e_loss.bacward()
				self.optimizer_e.step()	

				# all losses
				loss_dict = discriminator_loss_dict | generator_loss_dict | recon_loss_dict | perceptual_loss_dict | cycle_loss_dict

				# Logging related
				if self.global_step % self.args.wandb_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Log images of first batch to wandb
				if self.args.use_wandb and batch_idx == 0:
					self.wb_logger.log_images_to_wandb(x_2, y_2, y_2_hat, prefix="train", step=self.global_step, opts=self.args)

				# Validation related
				val_loss_dict = None
				if self.global_step % self.args.val_interval == 0 or self.global_step == self.args.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.args.max_steps:
					print('OMG, Holy Moly, Thank God, finished training!')
					break

				self.global_step += 1

	def set_train_status(self, train = True):
		if train:
			self.encoder.train()
			self.decoder.train()
			self.discriminator.train()
		else:
			self.encoder.eval()
			self.decoder.eval()
			self.discriminator.eval()


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

	def calc_loss(self, x=None, y_hat=None, latent=None, fake_pred=None, real_pred=None, w_fake=None, w_real=None, mean_path_length=None, loss_type=None):
		r"""
		loss_type is a list
		"""
		loss_dict = {}
		loss = 0.0
		types = ["adv_d", "adv_g", "reg", "rec_x", "lpips", "rec_w", "clf", "r1"]
		

		for curr_loss_name in loss_type:
			assert loss_type in types, "Invalid loss name"
			# adversarial losses
			#adv
			if curr_loss_name == "adv_d":
				loss_adv = self.adv_loss(real_pred, fake_pred, disc = True)
				loss_dict["adv_loss_d"] = loss_adv
				loss += self.args.lambdas["adv_d"] * loss_adv
			if curr_loss_name == "r1":
				loss_r1 = self.d_r1_loss(real_pred, x)
				loss_dict["r1_loss"] = loss_r1
				loss += self.args.lambdas["r1"] * loss_r1
			if curr_loss_name == "adv_g":
				loss_adv = self.adv_loss(real_pred, fake_pred, disc = False)
				loss_dict["adv_loss_g"] = loss_adv
				loss += self.args.lambdas["adv_g"] * loss_adv
			# path regularization
			if curr_loss_name == "reg":
				loss_reg, mean_path_length, path_lengths = self.reg_loss(y_hat, latent, mean_path_length)
				loss_dict["reg"] = loss_reg
				loss += self.args.lambdas["reg"] * loss_reg

			# reconstruction losses
			# rec_x
			if curr_loss_name == "rec_x":
				loss_rec_x = self.rec_x_loss(x, y_hat)
				loss_dict["rec_x"] = loss_rec_x
				loss += self.args.lambdas["rec_x"] * loss_rec_x
			# lpips
			if curr_loss_name == "lpips":
				loss_lpips = self.lpips_loss(x, y_hat)
				loss_dict["lpips"] = loss_lpips
				loss += self.args.lambdas["lpips"] * loss_lpips
			# rec_w
			if curr_loss_name == "rec_w":
				loss_rec_w = self.rec_w_loss(w_fake, w_real)
				loss_dict["rec_w"] = loss_rec_w
				loss += self.args.lambdas["rec_w"] * loss_rec_w
			# clf
			if curr_loss_name == "clf":
				loss_clf = self.clf_loss(x, y_hat) 
				loss_dict["clf"] = loss_clf
				loss += self.args.lambdas["clf"] * loss_clf

		loss_dict['loss'] = float(loss)
		return loss, loss_dict, mean_path_length
	
	def validate(self):

		self.set_train_status(train = False)
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):

			x_all, y_all = batch["inputs"], batch["labels"]

			with torch.no_grad():
				# during validation only use the autoencoder branch 
				x_2, y_2 = x_all[1], y_all[1]
				x_2, y_2 = x_2.to(self.device).float(), y_2.to(self.device).float()

				# get conditioning
				conditioning_2 = self.classifier(x_2)	

				# get encodings
				E_2 = self.encoder(x_2)

				# get output of generator	
				y_2_hat, latent_2 = self.generator(style = E_2, 
												   conditioning = conditioning_2, 
												   use_style_encoder = False, 
												   return_latents = True)

				w_fake_2 = self.encoder(y_2_hat)
				
				# calculate losses
				which_loss = ["rec_x", "rec_w"]
				recon_loss, recon_loss_dict, _ = self.calc_loss(x = x_2,
											y_hat = y_2_hat,
											w_fake = w_fake_2,
											w_real = E_2,
											loss_type = which_loss)
				
				which_loss = ["lpips"]
				perceptual_loss, perceptual_loss_dict, _ = self.calc_loss(x = x_2, 
												 y_hat = y_2_hat,
											     loss_type = which_loss)

				which_loss = ["clf"]
				cycle_loss, cycle_loss_dict, _ = self.calc_loss(x = x_2,
											y_hat = y_2_hat,
											loss_type = which_loss)

				# combine losses
				loss_dict = recon_loss_dict | perceptual_loss_dict | cycle_loss_dict

			agg_loss_dict.append(loss_dict)

			# Logging related
			# TODO Pick up here
			self.parse_and_log_images(x_2, y_2, y_2_hat,
									  title='images/test/dogs-cats',
									  subscript='{:04d}'.format(batch_idx))

			# Log images of first batch to wandb
			if self.args.use_wandb and batch_idx == 0:
				self.wb_logger.log_images_to_wandb(x, y, y_hat, prefix="test", step=self.global_step, opts=self.args)

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.set_train_status(train = True)
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

	def parse_and_log_images(self, x, y, y_hat, title, subscript=None, display_count=2):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input': common.tensor2im(x[i], self.opts),
				'target': common.tensor2im(y[i]),
				'output': common.tensor2im(y_hat[i]),
			}
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

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
			'state_dict': self.net.state_dict(),
			'generator_state_dict': self.net.decoder.state_dict(),
			'discriminator_state_dict': self.net.discriminator.state_dict(),
			'encoder_state_dict': self.net.encoder.state_dict(),
			'opts': vars(self.args)
		}
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.args.start_from_latent_avg:
			save_dict['latent_avg'] = self.net.latent_avg
		return save_dict