#!/bin/bash

python3 run_venom_uae.py --images_root "../../datasets/imagenet_compatible/imagenet-compatible/images" \
 --label_path "../../datasets/imagenet_compatible/imagenet-compatible/labels.txt" \
 --beta 0.5 \
 --save_dir "./out/venom_uae" 