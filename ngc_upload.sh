#!/bin/bash


ngc workspace upload \
    --source .  --destination /  \
    --exclude "*/.git/*" --exclude "*/configs/*" \
    --exclude "*/MNIST/*" --exclude "*/data/*"\
    --exclude "*/logger/*" --exclude "*/logs/*" \
    --exclude "*/val_fldr-Transformer/*" \
    --exclude "*/val_fldr-Transformer_MNIST_cont_256/*" \
    --exclude "*/wandb/*" \
    --exclude "*/third_party/*" \
    --exclude "*/lightning_logs/*" --exclude "*/saved_models/*" \
    --exclude "*.pyc" --exclude "*.egg" --exclude "*.pdf" --exclude "*.gv"  --exclude "*.mat" --exclude "*.o" --exclude "*.so"\
    --exclude "*.tar" --exclude "*.pth" --exclude "*.pt" \
    astro_code 
