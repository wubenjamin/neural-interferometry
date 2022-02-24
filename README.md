# Neural Interferometry: Image Reconstruction from Astronomical Interferometers using Transformer-Conditioned Neural Fields

Benjamin Wu,<sup>1,2*</sup> Chao Liu,<sup>2</sup> Benjamin Eckart,<sup>2</sup> Jan Kautz<sup>2</sup>

<sup>1</sup> National Astronomical Observatory of Japan <sup>2</sup> NVIDIA

<sup>*</sup> Work done as part of NVIDIA AI Residency program

### Abstract
Astronomical interferometry enables a collection of telescopes to achieve angular resolutions comparable to that of a single, much larger telescope. This is achieved by combining simultaneous observations from pairs of telescopes such that the signal is mathematically equivalent to sampling the Fourier domain of the object. However, reconstructing images from such sparse sampling is a challenging and ill-posed problem, with current methods requiring precise tuning of parameters and manual, iterative cleaning by experts. We present a novel deep learning approach in which the representation in the Fourier domain of an astronomical source is learned implicitly using a neural field representation. Data-driven priors can be added through a transformer encoder. Results on synthetically observed galaxies show that transformer-conditioned neural fields can successfully reconstruct astronomical observations even when the number of visibilities is very sparse.

### Run the demo

#### setup the conda environment
Set up the conda environment using the `requirements.txt` file:
```
conda create --name <env> --file requirements.txt
```
Please replace `<env>` with any name for the environment you like.

#### download the dataset
Download the dataset [here](https://drive.google.com/drive/folders/1d53MuR8KINIrVPJPTI7eP918dg_ZetVI?usp=sharing)

#### download the pretrained model
Download the model [here](https://drive.google.com/drive/folders/11eC0cQEi7gLAO6VOLp9lcqyFjFvkSyWA?usp=sharing)


#### inference using the pretrained model
Simply run the `eval_model.sh` script from command line:
```
sh ./eval_model.sh
```
To run this, you will need to modify the model, datapath path parameter within the bash script.
The script will load the pre-trained model and perform the inference on the test dataset. The results
would be saved in the `'.../test_res'` folder as images.
