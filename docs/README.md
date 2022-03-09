# styleSpaceAnalysis

Download the [afhq dataset](https://www.kaggle.com/andrewmvd/animal-faces) and extract the files in the root directory (if you choose to change the save location of the dataset, make sure to update the directories in the files appropriately). 

Run standalone_pipeline.ipynb to train a dogs vs cat classifier. If you want to use weights from our training, they can be found in the checkpoints directory. 

Weights for the classifer can be found at: `/data/vision/polina/scratch/avaidya/styleSpaceAnalysis/checkpoints`

# Experiments 
- baseline: lr_g:1e-3, lr_d:1e-3, lr_e1e-3:, weights:{"adv_d":1.,"adv_g":1., "reg":1., "rec_x":1., "rec_w":1., "lpips":1., "clf":1., "r1" : 5}, img_size:128, latent_dim:256, batch_size:2, channel_multiplier:1
