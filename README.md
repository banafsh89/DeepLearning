# DeepLearning
This is an implementation of MaskRCNN network that is used to find cluster structures. The implementations is done in the cluster.py script.
I have used the MaskRCNN network presented in this git repository: https://github.com/matterport/Mask_RCNN/tree/master and implemented the following changes to make it compatible with our data set and have good results.

  * Item I optimized the training schedule to train different stages of the network with optimized learning rate, epochs, and augmentation.
  * Item Multiple image augmentation techniques.
  *  Item Optimized learning momentum, weight decay, pooling size, and other parameters of the network.

To train the network I labeled the data by hand using the image editing tool, Pixelmator.

