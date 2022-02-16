from utils import def_transforms, def_datasets, def_dataloaders, visualize_model, train_and_val_model
from model import clf
import torch.optim as optim
from torch.optim import lr_scheduler
import sys
from argparse import Namespace
import wandb
wandb.init(project="cat-dog-styleSpace", entity="stylespace")


sys.path.append("./")
sys.path.append("../")

def main():

    args = Namespace(device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
                 train_dir = "../data/afhq/train",
                 val_dir = "../data/afhq/val",
                 save_path = "./checkpoints",
                 seed = 7,
                 labels = ["cat", "dog"],
                 batch_size = 64,
                 epochs = 50,
                 num_workers = 0,
                 class_names = {0:"cat", 1:"dog"} ,
                 lr = 0.0001,
                 momentum = 0.9,
                 criterion = nn.CrossEntropyLoss(),
                 optimizer = "SGD",
                 scheduler = "STEP",
                 scheduler_step_size = 7,
                 scheduler_gamma = 0.1,
    )


    # define transforms
    train_transforms, val_transforms = def_transforms()

    # define datasets and sizes
    datasets, dataset_sizes = def_datasets(args, train_transforms, val_transforms)

    # define dataloaders
    dataloaders = def_dataloaders(args, datasets["train"], datasets["val"])
    
    # define model
    model = clf(len(args.labels))
    model = model.to(args.device)
    
    # define criterion
    criterion = args.criterion
    
    # define optim
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    # define lr scheduler
    if args.scheduler == "STEP":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    # logging
    wandb.config = {
                    "learning_rate": args.lr,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size
    }
    
    # train and val model
    model_final = train_and_val_model(model, datasets, dataloaders, args.device, 
                                     criterion, optimizer, scheduler, args.save_path, args.epochs)

    visualize_model(model_final, dataloaders, args.device, args.class_names)

if __name__ == "__main__":
    main()