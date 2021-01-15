# WGAN-GP (TF2)

## Description
Reimplementation of HitLuca (Luca Simonetto)'s Master's thesis project in Tensorflow 2. (First described in https://esc.fnwi.uva.nl/thesis/centraal/files/f1376208007.pdf)

Warning: I do not have access to a GPU server, and as such my only available method for testing is through Tensorflow. As such, I am unable to replicate the testing conditions from the original paper.

Current progress:

Reimplement models in TF2:
 - WGAN: TODO
 - WGAN-GP: DONE
 - VAE: TODO
  
Reimplement helper code in TF2/updated libraries:
 - utils: IN PROGRESS

## Project structure

The ```master_thesis/generative_models``` folder contains the generative models used, along with the scripts for metrics calculation in ```master_thesis/comparison_metrics```.

The ```datasets``` folder contains the datasets used (at the moment only the [Berka dataset](https://sorry.vse.cz/~berka/challenge/pkdd1999/berka.htm)


Results are saved into the ```outputs``` folders, divided into subfolders for each model.

```utils.py``` contains meta-variables used at training time.

## Getting started
This project is intended to be self-contained, with the only exception being the dataset that is downloaded automatically.

Before starting, run the ```setup.py``` script, that will automatically download and parse the dataset, creating ready-to-use .npy files.


### Prerequisites
To install the python environment for this project, refer to the [Pipenv setup guide](https://pipenv.readthedocs.io/en/latest/basics/)

Currently identified requirements:

 - matplotlib
 - scipy==1.1.0
 - tensorflow
 - 

### Training the models
As the Java server is now integrated in the project, there is no need to start it separately.

The ```train_model.py``` file contains all the code necessary to train a generative model on the berka dataset. The model type is passed as an argument.

The outputs of the model are stored in the ```outputs/model_name``` folder.

### Comparing different models
First, move the output folder for each model in the ```comparison_metrics/comparison_datasets``` folder, then run the ```evaluate_datasets.py``` script in the ```comparison_metrics``` folder.
