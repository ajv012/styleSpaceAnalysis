import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))

def vis_outputs(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(8, 4 * display_count))
	gs = fig.add_gridspec(display_count, 3)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		plot_outputs(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig

def plot_outputs(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input'])
	plt.title('Input')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target'])
	plt.title('Target')
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output'])
	plt.title('Output')