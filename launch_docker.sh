#!/bin/bash
docker run \
    --ipc=host \
    --gpus all \
    -v "/home/chaoliu/astroImaging/Transformer_encoder:/code" \
    -ti nvcr.io/nvidian/lpr/cxl-astro:latest
    
