#!/bin/bash

#Image LOSS: Gal10-DEC #
python train_EHTTransEncoder.py \
--eval \
--exp_name  Galaxy10-DEC-cont/transformer/mlp_8_layer/image_loss-NF_128 \
--yaml_file ../../ngc_exp/Galaxy10-DEC-cont/transformer/mlp_8_layer/image_loss-NF_128/hparams.yaml \
--model_checkpoint ../../ngc_exp/Galaxy10-DEC-cont/transformer/mlp_8_layer/image_loss-NF_128/checkpoints/model.ckpt \
--loss_type image \
--num_fourier 128 \
--input_size 256 \
--dataset Galaxy10_DECals \
--data_path_cont ../../data/eht_cont_200im_Galaxy10_DECals_full.h5 \
--data_path_imgs ../../data/Galaxy10_DECals.h5 \
--dataset_path   ../../data/eht_grid_128FC_200im_Galaxy10_DECals_full.h5 \
