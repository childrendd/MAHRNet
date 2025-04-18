# Enhanced High-Resolution Multi-Attention Network for Precise Spinal Image Segmentation

# Datasets
* The SpineSagT2Wdataset3 dataset at https://pan.baidu.com/s/1nSFzRdn6FNnbR2YkpfTwsQ, the extraction code is 5678. The dataset contains a total of 195 patients' spine images, which were sliced into a total of 2460 images and their corresponding masks.
* The Verse2020 dataset at https://pan.baidu.com/s/1Pkii45q6iudLfBetMwo1Jw,  the extraction code is 1234. The dataset contains a total of 54 patients' spine images, which were sliced into a total of 1723 images and their corresponding masks. 
* The Spine1K dataset at https://pan.baidu.com/s/1h5tvvqEPknYX5R2iX1GIFw, the extraction code is 2580. Due to the large size of this dataset, we selected a total of 225 patients' spine images, which were sliced into a total of png images and their corresponding masks.

# Introduction
* If you want to use our code, you must have the following preparation under the PyTorch framework: see requirement.txt for details. 
* Code Guidance: Download the dataset in the above link, put the training images and labels into "VOCdevkit/VOC2007/JPEGImages" and "VOCdevkit/VOC2007/SegmentationClass", then run Data_Split.py file to divide the data set can be seen in the “VOCdevkit/VOC2007/ImageSets/Segmentation out of the training and validation of the document, then you can run the train.py file for training, training is complete run get_miou.py After training, you can run get_miou.py to test the file.
* Running the get_miou.py file will generate prediction results with pixel values of 0 and 1. Since these pixel values are relatively small, the prediction images appear with poor contrast when saved in PNG format (which is why they look almost entirely black). This is normal. To address this, you can run the adjust.py file to perform binarization processing on the prediction images.
* When training with datasets outside of the aforementioned ones, you will first need to convert the dataset into the format required by the network. Specifically, before training, you must use the bit.py file to transform the masks into the correct format.

# Reference
* Li, H., Ren, Z., Zhu, G. et al. Enhancing medical image segmentation with MA-UNet: a multi-scale attention framework. The Visual Computer (2025).
