import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import time, copy

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import wandb

from dataset import afhq_dataset


def train_and_val_model(args, model, datasets, dataloaders, device, criterion, optimizer, scheduler, PATH, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss_train = 0.0
        running_corrects_train = 0

        running_loss_val = 0.0
        running_corrects_val = 0

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Iterate over data.
            for batch in dataloaders[phase]:
                inputs, labels = batch["inputs"], batch["labels"]
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                if phase == "train":
                    running_loss_train += loss.item() * inputs.size(0)
                    running_corrects_train += torch.sum(preds == labels.data)
                else:
                    running_loss_val += loss.item() * inputs.size(0)
                    running_corrects_val += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            if phase == "train":
                epoch_loss = running_loss_train / len(datasets[phase])
                epoch_acc = running_corrects_train.double() / len(datasets[phase])
                wandb.log({"train_epoch_loss": epoch_loss, "train_epoch_acc": epoch_acc})
            else:
                epoch_loss = running_loss_val / len(datasets[phase])
                epoch_acc = running_corrects_val.double() / len(datasets[phase])
                wandb.log({"val_epoch_loss": epoch_loss, "val_epoch_acc": epoch_acc})

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # save current best model
                PATH = "{}/checkpoint_{}.pt".format(args.save_path, epoch)
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': epoch_loss,
                            'acc' : epoch_acc,
                            }, PATH)
                wandb.log({"best_acc": best_acc})

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, dataloaders, device, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for batch in enumerate(dataloaders['val']):
            inputs, labels = batch["inputs"], batch["labels"]
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                plt.imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def def_transforms():
    train_transforms = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transforms, val_transforms

def def_datasets(args, train_transforms, val_transforms):
    dataset_train = afhq_dataset(args.train_dir, args.seed, args.labels, train_transforms)
    dataset_val = afhq_dataset(args.val_dir, args.seed, args.labels, val_transforms)
    datasets = {"train": dataset_train, "val": dataset_val}
    dataset_sizes = {x: datasets[x] for x in ['train', 'val']}

    return datasets, dataset_sizes

def def_dataloaders(args, dataset_train, dataset_val):
    dataloader_train = DataLoader(dataset_train, batch_size = args.batch_size, num_workers=args.num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size = args.batch_size, num_workers=args.num_workers)
    dataloaders = {"train": dataloader_train, "val":dataloader_val}

    return dataloaders