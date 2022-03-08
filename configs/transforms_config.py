from abc import abstractmethod
import torchvision.transforms as transforms


class TransformsConfig(object):

	def __init__(self, opts):
		self.opts = opts

	@abstractmethod
	def get_transforms(self):
		pass


class afhq_Transforms(TransformsConfig):

	def __init__(self, args):
		super(afhq_Transforms, self).__init__(args)

	def get_transforms(self):
		transforms_dict = {
			'transform_train': transforms.Compose([
				transforms.Resize((args.img_size)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_val': transforms.Compose([
				transforms.Resize((args.img_size)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((args.img_size)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict