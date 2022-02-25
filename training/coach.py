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
from criteria import adv_loss, clf_loss, path_reg_loss
from criteria.lpips.lpips import LPIPS
from configs import data_configs
from datasets.afhq_dataset import afhq_dataset

from utils.wandb_utils import WBLogger

from models.net import net
from training.ranger import Ranger


class Coach:
	def __init__(self, args):
		self.args = args

		self.global_step = 0

		# TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES (use distributed module)
		self.device = self.args.device  

		if self.args.use_wandb:
			self.wb_logger = WBLogger(self.args)
		
		# Initialize network
		self.net = net(self.args).to(self.device)

		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.net.latent_avg is None:
			self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()

		# Initialize loss
		# adv loss
		if self.args.lambdas["adv"] > 0:
			# adv loss requires generative part as well 
			self.adv_loss = adv_loss().to(self.device) 
		# path regularization
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

		# Initialize optimizer
		self.optimizer_g, self.optimizer_d = self.configure_optimizers()

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

	def configure_optimizers(self):
		# encoder + decoder optim
		params_g = list(self.net.encoder.parameters())
		if self.args.train_decoder:
			params_g += list(self.net.decoder.parameters())
		if self.args.optim_name == 'adam':
			g_optimizer = torch.optim.Adam(params_g, lr=self.args.lr)
		else:
			g_optimizer = Ranger(params_g, lr=self.args.lr)

		# discriminator optim
		params_d = self.net.discriminator.parameters()
		d_optimizer = optim.Adam(params_d, lr=self.args.lr_d)

		return g_optimizer, d_optimizer

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
		self.net.train()
		mean_path_length = 0 
		while self.global_step < self.args.max_steps:

			for batch_idx, batch in enumerate(self.train_dataloader):

				x, y = batch["inputs"], batch["labels"]
				x, y = x.to(self.device).float(), y.to(self.device).float()

				self.optimizer_g.zero_grad()
				self.optimizer_d.zero_grad()

				# get model outputs
				y_hat, latent = self.net.forward(x, return_latents=True)
				fake_pred = self.net.discriminator(y_hat)
				real_pred = self.net.discriminator(x) 

				self.net.encoder.eval()
				with torch.no_grad():
					w_fake = self.net.get_encodings(y_hat)
					w_real = self.net.get_encodings(x)
				self.net.encoder.train()
				
				########## Discriminator training ##########
				self.requires_grad(self.net.decoder, False)
				self.requires_grad(self.net.discriminator, True)
			
				# get discriminator loss
				discriminator_loss, adv_loss_dict, _ = self.calc_loss(x, y, y_hat, latent, fake_pred, real_pred, 
														  w_fake, w_real, mean_path_length, loss_type=["adv_d"])	

				# discriminator updates
				self.net.discriminator.zero_grad()	
				discriminator_loss.backward() # which model??
				self.optimizer_d.step()		

				### TODO: Pick up here
				# different optims for encoder and decoder?

				########## Generator ##########
				self.requires_grad(self.net.decoder, True)
				self.requires_grad(self.net.discriminator, False)
				
				# calculate losses
				which_loss = ["adv_g", "reg", "rec_x", "lpips", "rec_w", "clf"]
				loss, loss_dict, mean_path_length = self.calc_loss(x, y, y_hat, latent, fake_pred, real_pred,
												 w_fake, w_real, mean_path_length, loss_type=which_loss)

				# backward and step
				loss.backward()
				self.net.zero_grad()
				self.optimizer_g.step()

				# all losses 
				loss_dict = adv_loss_dict | loss_dict

				# Logging related
				if self.global_step % self.args.wandb_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Log images of first batch to wandb
				if self.args.use_wandb and batch_idx == 0:
					self.wb_logger.log_images_to_wandb(x, y, y_hat, prefix="train", step=self.global_step, opts=self.args)

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

	def calc_loss(self, x, y, y_hat, latent, fake_pred, real_pred, w_fake, w_real, mean_path_length, loss_type):
		r"""
		loss_type is a list
		"""
		loss_dict = {}
		loss = 0.0
		types = ["adv_d", "adv_g", "reg", "rec_x", "lpips", "rec_w", "clf"]
		

		for curr_loss_name in loss_type:
			assert loss_type in types, "Invalid loss name"
			# adversarial loss
			if curr_loss_name == "adv_d":
				loss_adv = self.adv_loss(real_pred, fake_pred, disc = True)
				loss_dict["adv_loss_d"] = loss_adv
				loss += self.args.lambdas["adv_d"] * loss_adv
			if curr_loss_name == "adv_g":
				loss_adv = self.adv_loss(real_pred, fake_pred, disc = False)
				loss_dict["adv_loss_g"] = loss_adv
				loss += self.args.lambdas["adv_g"] * loss_adv
			# path regularization
			if curr_loss_name == "reg":
				loss_reg, mean_path_length, path_lengths = self.reg_loss(y_hat, latent, mean_path_length)
				loss_dict["reg"] = loss_reg
				loss += self.args.lambdas["reg"] * loss_reg
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
		self.net.eval()
		self.net.discriminator.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):

			x, y = batch["inputs"], batch["labels"]

			with torch.no_grad():
				x, y = x.to(self.device).float(), y.to(self.device).float()

				# get fake img, return_latent = False
				y_hat = self.net.decoder(x)

				# get discriminator outputs
				fake_pred = self.net.discriminator(y_hat)
				real_pred = self.net.discriminator(x) 

				# generator output
				y_hat, latent = self.net.forward(x, return_latents=True)

				
				w_fake = self.net.get_encodings(y_hat)
				w_real = self.net.get_encodings(x)

				# Tget losses
				which_loss = ["adv", "reg", "rec_x", "lpips", "rec_w", "clf"]
				loss, loss_dict, mean_path_length = self.calc_loss(x, y, y_hat, latent, fake_pred, real_pred,
												 w_fake, w_real, mean_path_length, loss_type=which_loss)

			agg_loss_dict.append(loss_dict)

			# Logging related
			self.parse_and_log_images(x, y, y_hat,
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

		self.net.train()
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