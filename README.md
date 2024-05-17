# Low Emission Nebulae Image Detector (LENID)
LENID is a tool for the automated detection of low-emission extended objects in astronomical images, applying some techniques for image segmentation: image augmentation and Convolutional Neural Networks (CNN).

With this tool, we analyzed images from the INT Photometric H-Alpha Survey (IPHAS), an astronomical study from the northern plane of our galaxy (https://www.iphas.org/). They obtained it with the INT Wide Field Camera (WFC) at the Isaac Newton Telescope. The WFC has an array of four charge-coupled devices (CCD), each of which has 2K x 4K pixels.

The libraries used to implement the neural network were Keras, a high-level open-source neural network library capable of running in addition to popular deep learning platforms such as TensorFlow, Microsoft's CNTK, and Theano. Since we have a GPU from Nvidia, we use CUDA and cuDNN. CUDA, the NVIDIA Unified Computing Device Architecture (CUDA), is a parallel computing platform and API that enables an NVIDIA GPU to be used for general-purpose computing on graphics processing units (GPGPU).

CUDA Deep Neural Network (cuDNN) is a library of primitives for deep neural networks containing GPU-optimized implementations for routines such as convolutions, clustering, normalization, and trigger layers. It is part of the NVIDIA SDK for deep learning and supports the most common deep learning frameworks such as Caffe, CNTK, Pytorch, etc.

LENID it is divided into two modules: Cut to the stars, and image segmentation. 

The module cut to the stars considers the inconvenience of having bright stars and implies higher energy. As a result, it has high peaks with pixel values exceeding the maximum value allowed. 

The original maximum value for the images is 65,535, but we defined a threshold manually to set it to 1,000 since the background of the image averages the values of 50–80. An emission nebula can have ranges of values very close to the bottom, but it can also have areas with higher emissions and average values between 200 and 700. A simple threshold is then applied to set the pixels greater than the threshold as null and be able to replace them with the neighboring pixels.

The module image segmentation uses a modified U-NET architecture presented by Ronneberger in 2015, which is a model that has worked well for segmentation in biomedical images. The module is divided into three functions image augmentation, training CNN U-NET, and low emission nebulae detection. 

For now the is no user interface, it has to run with command line.

Usage:

- To specify the input directory of the images: -d or --dir_images
- To specify the output directory of the images: -r or --dir_results
- To specify that you want to train the neural network use: -t or --train
- To specify if you want to segment the test images use: -s or --segment and -o or --extended



