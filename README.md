### StrassenNets

This notebook gives step-by-step instructions for training Strassen-ResNet-20 on CIFAR-10 using [mxnet](https://github.com/apache/incubator-mxnet/) and [gluon](http://gluon.mxnet.io/), see Section 4.2.1. in the paper

[StrassenNets: Deep learning with a multiplication budget, M. Tschannen, A. Khanna, A. Anandkumar, 2017.](https://arxiv.org/abs/1712.03942)

#### Paper abstract

A large fraction of the arithmetic operations required to evaluate deep neural networks (DNNs) are due to matrix multiplications, both in convolutional and fully connected layers. Matrix multiplications can be cast as 2-layer sum-product networks (SPNs) (arithmetic circuits), disentangling multiplications and additions. We leverage this observation for end-to-end learning of low-cost (in terms of multiplications) approximations of linear operations in DNN layers. Specifically, we propose to replace matrix multiplication operations by SPNs, with widths corresponding to the budget of multiplications we want to allocate to each layer, and learning the edges of the SPNs from data. Experiments on CIFAR-10 and ImageNet show that this method applied to ResNet yields significantly higher accuracy than existing methods for a given multiplication budget, or leads to the same or higher accuracy compared to existing methods while using significantly fewer multiplications. Furthermore, our approach allows fine-grained control of the tradeoff between arithmetic complexity and accuracy of DNN models. Finally, we demonstrate that the proposed framework is able to rediscover Strassen's matrix multiplication algorithm, i.e., it can learn to multiply 2x2 matrices using only 7 multiplications instead of 8.


The notebook provides a detailed introduction to the method.
