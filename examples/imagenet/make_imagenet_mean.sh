#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/hiroki11x/ImageNet
DATA=/home/hiroki11x/ImageNet
TOOLS=/home/hiroki11x/dl/nvcaffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/ilsvrc12_train_lmdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
