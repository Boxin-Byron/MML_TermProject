#!/bin/bash
set -e

mkdir -p dataset/iiw400
mkdir -p dataset/docci
mkdir -p dataset/coco

echo "Downloading Docci Images..."
wget -c -P dataset/docci https://storage.googleapis.com/docci/data/docci_images.tar.gz

echo "Extracting Docci Images..."
tar -xzf dataset/docci/docci_images.tar.gz -C dataset/docci

echo "Downloading Docci AAR Images (referenced by IIW-400)..."
wget -c -P dataset/docci https://storage.googleapis.com/docci/data/docci_images_aar.tar.gz

echo "Extracting Docci AAR Images..."
tar -xzf dataset/docci/docci_images_aar.tar.gz -C dataset/docci

cd dataset/coco
# 下载验证集图片 (13GB)
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip

# 下载标注文件
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip