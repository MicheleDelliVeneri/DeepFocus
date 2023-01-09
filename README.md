
# Deep Focus
<img src="/icons/DeepFocus.png" width=25% height=25% style="float: left;margin-right: 2px;margin-top: 5px;">
Deep Focus, a metalearner for image deconvolution, source detection and characterization within radio interferometric data cubes.
Currently, Deep Focus is able to deconvolve simulated ALMA data cubes, detect and characterize sources within them or to perform a full deconvolution and source detection and characterization pipeline for SKA data cubes.
## Installation
- DeepFocus requires that you have NVIDIA Drivers installed, download latest from [NVIDIA](https://www.nvidia.com/download/index.  aspx). Also it requires that you have downloaded and installed conda. Get latest conda distribution from [CONDA](https://www.anaconda.com/products/distribution). After you have downloaded it, installed it and update it to the latest version by running the following commands in your terminal
`conda update conda`
`conda update --all`
-  Move inside the DeepFocus directory and use the requirements.txt to create a conda environment with all the packages needed to run DeepFocus
`conda create --name DeepFocusEnv --file requirements.txt`
this command will create  a conda environment named DeepFocus with inside already all the needed packages to run DeepFocus.
Activate the environment `conda activate DeepFocusEnv` and you are ready to go. 

## Instructions
Deep Focus can be used to solve 2D and 3D Deconvolution, Segmentation, Regression and Classification problems by creating and optimizing the architectures of respectively 2D and 3D CNN-based architectures. The type of problem to solve, and thus the type of architectures which are built, can be selected by changing the `dmode` parameter. 
- dmode = `deconvolver` will build architectures whose output is an image or a cube such as U-Nets and Convolutional Autoencoders. The number of input and output channels are controlled by the `in_channels` and `out_channels` parameters. These can be different, for example `in_channels` could assume a value greater than one if more than one band is fed to an architecture or if different cubes are used as different channels (continuum, hi, moment-masked and so on). 
- dmode = `regressor` will build architectures for parameter regression such as DenseNets, ResNets and so on. The size of the parametric vector to regress and thus the number of parameters which are predicted is controlled by the output channels parameter. 
- dmode = `classifier` will build architectures for image or cube classification. The number of classes substitutes the number of output channels. 
DeepFocus assumes that your data is in .fits format. In particular it requires that your data is arleady divided in Train, Validation and Test set with the following directory structure:
- Train
    - Cube1.fits
    - Cube2.fits
    - ....
    - train_params.csv
- Test
    - Cube3.fits
    - Cube4.fits
    - test_params.csv
- Validation
    - Cube5.fits
    - Cube6.fits
    - valid_params.csv

Where the params .csv files should contain for each source, the ID of the cube, the x, y, (z) coordinates of the source and the parameters to regress or classify.

### General Overview of the package
The package options such as the data, the model and its hyperparameters are all controlled through dictionaries. In particular, each file starts with a dictionary which should be filled by the user with their parameters of choise. 
Within the directory, there are three files:
- train_model.py is a routine to train a model with a given configuration and save its weights on th basis of  the validation loss improvenment over time;
- test_model.py is a routine to test a model with a given configuration and produce predictions on the Test data and related plots;
- sweep.py is a routine to perform model and hyperparameter optimization, while tecnically these could be optimized at the same time, we advise to first seach for the best architecture, and then to optimize its hyperparameters (learning rate, batch size, dropout, weight decay and so on).
### Architectures
The package is capable of building several Deeep Learning 2D and 3D architectures such us:
- Convolutional Autoencoder (CAE)
- U-Net
- ResNets (18, 34, 56, 121, .....)
- DenseNet
- VGGNet

The architectures are built using the PyTorch library through the following parameters:
- `in_channels` (scalar) is the number of input channels, for example if you have a cube with 3 bands, you should set `in_channels = 3`
- `out_channels` (scalar) is the number of output channels, for example if you want to regress 3 parameters, you should set `out_channels = 3`
- `depths`: (list) is the depth of the convolutional blocks that make the architecture. For example, if you want to build a U-Net with 3 convolutional blocks, each one containing
            two repeated convolutions for each block,  you should set `depths = [2, 2, 2]`. 
            If you want to build a ResNet (18) with 4 convolutional blocks, each one containing 2 repeated convolutions for each block, you should set `depths = [2, 2, 2, 2]`. If you want to build a DenseNet with 4 convolutional blocks, each one containing 2 repeated convolutions for each block, you should set `depths = [2, 2, 2, 2]`. If you want to build a VGGNet with 4 convolutional blocks, each one containing 2 repeated convolutions for each block, you should set `depths = [2, 2, 2, 2]`. If you want to build a CAE with 4 convolutional blocks, each one containing 2 repeated convolutions for each block, you should set `depths = [2, 2, 2, 2]`.
            If you want to build a ResNet 34, depths should be set to `[3, 4, 6, 3]` and so on.
- `block`: (string), it can take the values `basic` or `bottleneck`. If you want to build a ResNet with bottleneck layers, you should set `block = 'basic'` 
            and if you want to build a DenseNet, you should set `block = 'bottleneck'`. Bottleneck layers are used to reduce the number of parameters in the network, by increasing and decreasing it within the block. 
- `growth_rate`: (scalar) is the growth rate of the DenseNet and ResNets. If you want to build a DenseNet with a growth rate of 12, you should set `growth_rate = 12`, the default value for a ResNet is 4.
- `skip-connections`: (boolean) if you want to build a ResNet with skip connections, you should set `skip-connections = True`, the default value is `False`. Skip connections are used   to increase the gradient flow in the network, and thus to improve the training.
- `dropout`: (scalar) is the dropout rate, if you want to build an architecture with a dropout rate of 0.2, you should set `dropout = 0.2`, the default value is 0.0. The Dropout is only used within the last fully connected layers. 
- `block_sizes` (list): the number of channels for each layer (which could be composed by multiple blocks) of a given Architecture. For example, if you want to build a Network with 3 layers, the first creating 16 channles, 
the second 32 and the last 128 you should set  `block_sizes = [32, 64, 128]`.
-


## Missing Features still to be transfered
- parameter optimization routines
- output routines and plots



### Cite us
Michele Delli Veneri, Łukasz Tychoniec, Fabrizia Guglielmetti, Giuseppe Longo, Eric Villard, 3D Detection and Characterisation of ALMA Sources through Deep Learning, Monthly Notices of the Royal Astronomical Society, 2022;, stac3314, https://doi.org/10.1093/mnras/stac3314

@article{10.1093/mnras/stac3314,
    author = {Delli Veneri, Michele and Tychoniec, Łukasz and Guglielmetti, Fabrizia and Longo, Giuseppe and Villard, Eric},
    title = "{3D Detection and Characterisation of ALMA Sources through Deep Learning}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    year = {2022},
    month = {11},
    issn = {0035-8711},
    doi = {10.1093/mnras/stac3314},
    url = {https://doi.org/10.1093/mnras/stac3314},
    note = {stac3314},
    eprint = {https://academic.oup.com/mnras/advance-article-pdf/doi/10.1093/mnras/stac3314/47014718/stac3314.pdf},
}
