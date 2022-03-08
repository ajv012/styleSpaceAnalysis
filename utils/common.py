import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def tensor2im(var):
	var = var.cpu().detach().numpy().transpose(1, 2, 0)
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))

def vis_outputs(log_hooks):
	r"""
	log_hooks is a list of dictionaries. Keys in dictionary are input and output. Values are PIL images
	"""
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(8, 4 * display_count))
	gs = fig.add_gridspec(display_count, 2)
	for i in range(display_count):
		hooks_dict = log_hooks[i] # dictionary has input, output, and title info
		fig.add_subplot(gs[i, 0])
		plot_outputs(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig

def plot_outputs(hooks_dict, fig, gs, i):
	
	title_info = hooks_dict["title_info"]
	# use dict title_info in hooks dict to construct titles
	input_title = " true label = {}\n pred label = {}\n top class score = {:.2f}".format(
		title_info["true_label"].detach().cpu().numpy(),
		title_info["pred_label_x"].detach().cpu().numpy()[0],
		title_info["top_score_x"].detach().cpu().numpy()[0]
	)

	output_title = " true label = {}\n pred label = {}\n top class score = {:.2f}".format(
		title_info["true_label"].detach().cpu().numpy(),
		title_info["pred_label_y_hat"].detach().cpu().numpy()[0],
		title_info["top_score_y_hat"].detach().cpu().numpy()[0]
	)
	plt.imshow(hooks_dict['input'])
	plt.title(input_title)
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['output'])
	plt.title(output_title)