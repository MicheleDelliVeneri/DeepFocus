
# Deep Focus


Deep Focus    | badges
--------------| --------------
<img src="/icons/DeepFocus.png" width=50% height=50%> | Deep Focus, a metalearner for image deconvolution, source detection and characterization within radio interferometric data cubes.



## instructions
TBW

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
    abstract = "{We present a Deep-Learning (DL) pipeline developed for the detection and characterization of astronomical sources within simulated Atacama Large Millimeter/submillimeter Array (ALMA) data cubes. The pipeline is composed of six DL models: a Convolutional Autoencoder for source detection within the spatial domain of the integrated data cubes, a Recurrent Neural Network (RNN) for denoising and peak detection within the frequency domain, and four Residual Neural Networks (ResNets) for source characterization. The combination of spatial and frequency information improves completeness while decreasing spurious signal detection. To train and test the pipeline, we developed a simulation algorithm able to generate realistic ALMA observations, i.e. both sky model and dirty cubes. The algorithm simulates always a central source surrounded by fainter ones scattered within the cube. Some sources were spatially superimposed in order to test the pipeline deblending capabilities. The detection performances of the pipeline were compared to those of other methods and significant improvements in performances were achieved. Source morphologies are detected with subpixel accuracies obtaining mean residual errors of 10−3 pixel (0.1 mas) and 10−1 mJy/beam on positions and flux estimations, respectively. Projection angles and flux densities are also recovered within \\$10\\\\%\\$ of the true values for \\$80\\\\%\\$ and \\$73\\\\%\\$ of all sources in the test set, respectively. While our pipeline is fine-tuned for ALMA data, the technique is applicable to other interferometric observatories, as SKA, LOFAR, VLBI, and VLTI.}",
    issn = {0035-8711},
    doi = {10.1093/mnras/stac3314},
    url = {https://doi.org/10.1093/mnras/stac3314},
    note = {stac3314},
    eprint = {https://academic.oup.com/mnras/advance-article-pdf/doi/10.1093/mnras/stac3314/47014718/stac3314.pdf},
}
