import math
import torch
from torch import nn

from models.encoders import encoders
from models.stylegan2.model import Generator
from models.discriminator.model import Discriminator
from models.classifier import Classifier

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class net(nn.Module):

	def __init__(self, args):
		super(net, self).__init__()
		self.args = args

		# compute number of style inputs based on the output resolution
		self.args.n_styles = int(math.log(self.args.output_size, 2)) * 2 - 2

		# Define architecture
		self.encoder = self.set_encoder()
		
		# define generator 
		self.decoder = Generator(self.opts.output_size, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

		# define discrmiinator
		self.discriminator = Discriminator(self.args.img_size, self.channel_multiplier)

		# define the classifier 
		self.classifier = Classifier(self.args)

		self.latent_avg = None

		# Load weights if needed -> we are going to train from scratch
		# self.load_weights()

	def set_encoder(self):
		if self.opts.encoder == 'gradual':
			encoder = encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		
		return encoder

	# def load_weights(self):
	# 	if self.opts.checkpoint_path is not None:
	# 		print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
	# 		ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
	# 		self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
	# 		self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
	# 		self.__load_latent_avg(ckpt)
	# 	else:
	# 		print('Loading encoders weights from irse50!')
	# 		encoder_ckpt = torch.load(model_paths['ir_se50'])
	# 		# if input to encoder is not an RGB image, do not load the input layer weights
	# 		if self.opts.label_nc != 0:
	# 			encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
	# 		self.encoder.load_state_dict(encoder_ckpt, strict=False)
	# 		print('Loading decoder weights from pretrained!')
	# 		ckpt = torch.load(self.opts.stylegan_weights)
	# 		self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
	# 		if self.opts.learn_in_w:
	# 			self.__load_latent_avg(ckpt, repeat=1)
	# 		else:
	# 			self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# normalize with respect to the center of an average face
			# if True then will need pretrained weights
			if self.args.start_from_latent_avg:
				if self.args.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)


		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code

		### get clf output and concatenate it to the encoder output
		clf_out = self.classifier(x)
		codes = torch.cat([codes, clf_out]) # mostly wrong, but troubleshoot as you run the code. Need to know size of codes
		
		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, result_latent
		else:
			return images

	def get_encodings(self, x):
		r"""
		Get the encoding of x. Before coming here, encoder should be set to eval and no_grad should be used
		"""
		return self.encoder(x)

	# def __load_latent_avg(self, ckpt, repeat=None):
	# 	if 'latent_avg' in ckpt:
	# 		self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
	# 		if repeat is not None:
	# 			self.latent_avg = self.latent_avg.repeat(repeat, 1)
	# 	else:
	# 		self.latent_avg = None