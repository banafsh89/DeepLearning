# DeepLearning
This is an implementation of the MaskRCNN network, designed for identifying cluster structures. The implementation is contained within the 'cluster.py' script. I utilized the MaskRCNN network available in the following GitHub repository: https://github.com/matterport/Mask_RCNN/tree/master.

Here is an example of the structures we are looking for in an image. The red represents the ground truth, and the yellow represents the network prediction.



To adapt it for our dataset and achieve robust results, I introduced the following modifications:
  * Optimized Training Schedule: I fine-tuned the training schedule to optimize learning rates, epochs, and introduced data augmentation techniques at different stages of the network.
  * Multiple Image Augmentation Techniques: I incorporated various image augmentation techniques to enhance the model's robustness and ability to generalize.
  * Parameter Optimization: I optimized key parameters such as learning momentum, weight decay, pooling size, and other network-specific parameters to improve overall performance.

To prepare the network for training, I manually labeled the dataset using the image editing tool, Pixelmator. You can find the code used to create these labeled images in the 'CreateInputImages' folder.

The result of this study is shown in the following papers: https://arxiv.org/abs/2001.03116 and https://arxiv.org/abs/2012.04381.

