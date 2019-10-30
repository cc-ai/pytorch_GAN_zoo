# This is an Attempt to extend PGAN for Image to Image Translation
## Requirements

This project requires:
- pytorch
- torchvision
- numpy
- scipy
- h5py (fashionGen)

Optional:
- visdom
- nevergrad (inspirational generation)

## Installing the dependencies with pip
```
pip install -r requirements.txt
```

## Preparing a dataset
The datasets.py script allows you to prepare your datasets and build their corresponding configuration files.

### How to run a training session ?

```
python train.py PGAN -c $CONFIGURATION_FILE[-n $RUN_NAME][-d $OUTPUT_DIRECTORY][OVERRIDES]
```

Where:

1 - CONFIGURATION_FILE(mandatory): path to a training configuration file. This file is a json file containing at least a pathDB entry with the path to the training dataset. See below for more informations about this file.

2 - RUN_NAME is the name you want to give to your training session. All checkpoints will be saved in $OUTPUT_DIRECTORY/$RUN_NAME. Default value is default

3 - OUTPUT_DIRECTORY is the directory were all training sessions are saved. Default value is output_networks

4 - OVERRIDES: you can overrides some of the models parameters defined in "config" field of the configuration file(see below) in the command line. For example:


## Configuration file of a training session

The minimum necessary file for a training session is a json with the following lines

```
{
    "pathDB": PATH_TO_YOUR_DATASET
}
```

Where a dataset can be:
- a folder with all your images in .jpg, .png or .npy format
- a folder with N subfolder and images in it
- a .h5 file(cf fashionGen)

To this you can add a "config" entry giving overrides to the standard configuration. See models/trainer/standard_configurations to see all possible options. For example:

```
{
    "pathDB": PATH_TO_YOUR_DATASET,
    "config": {"baseLearningRate": 0.1,
               "miniBatchSize": 22}
}
```

Will override the learning rate and the mini-batch-size. Please note that if you specify a - -baseLearningRate option in your command line, the command line will prevail. Depending on how you work you might prefer to have specific configuration files for each run or only rely on one configuration file and input your training parameters via the command line.

Other fields are available on the configuration file, like:
- pathAttribDict(string): path to a .json file matching each image with its attributes. To be more precise with a standard dataset, it is a dictionary with the following entries:

```
{
    image_name1.jpg: {attribute1: label, attribute2, label ...}
    image_name2.jpg: {attribute1: label, attribute2, label ...}
    ...
}
```

With a dataset in the fashionGen format(.h5) it's a dictionary summing up statistics on the class to be sampled.

- imagefolderDataset(bool): set to true to handle datasets in the torchvision.datasets.ImageFolder format
- selectedAttributes(list): if specified, learn only the given attributes during the training session
- pathPartition(string): path to a partition of the training dataset
- partitionValue(string): if pathPartition is specified, name of the partition to choose
- miniBatchScheduler(dictionary): dictionary updating the size of the mini batch at different scale of the training
ex {"2": 16, "7": 8} meaning that the mini batch size will be 16 from scale 16 to 6 and 8 from scale 7
- configScheduler(dictionary): dictionary updating the model configuration at different scale of the training
ex {"2": {"baseLearningRate": 0.1, "epsilonD": 1}} meaning that the learning rate and epsilonD will be updated to 0.1 and 1 from scale 2 and beyond

## How to run a evaluation of the results of your training session ?

You need to use the eval.py script.

### Image generation

You can generate more images from an existing checkpoint using:
```
python eval.py visualization -n PGAN -m $modelType
```

Where modelType is PGAN and modelName is the name given to your model. This script will load the last checkpoint detected at testNets/$modelName. If you want to load a specific iteration, please call:

```
python eval.py visualization -n PGAN -m $modelType -s $SCALE -i $ITER
```

### Fake dataset generation

To save a randomly generated fake dataset from a checkpoint please use:

```
python eval.py visualization -n $modelName -m $modelType --save_dataset $PATH_TO_THE_OUTPUT_DATASET --size_dataset $SIZE_OF_THE_OUTPUT
```

### SWD metric

Using the same kind of configuration file as above, just launch:

```
python eval.py laplacian_SWD -c $CONFIGURATION_FILE -n $modelName -m $modelType
```

Where $CONFIGURATION_FILE is the training configuration file called by train.py (see above): it must contains a "pathDB" field pointing to path to the dataset's directory. For example, if you followed the instruction of the Quick Training section to launch a training session on celebaHQ your configuration file will be config_celebaHQ.json.

You can add optional arguments:

- -s $SCALE: specify the scale at which the evaluation should be done(if not set, will take the highest one)
- -i $ITER: specify the iteration to evaluate(if not set, will take the highest one)
- --selfNoise: returns the typical noise of the SWD distance for each resolution

### Inspirational generation

To make an inspirational generation, you first need to build a feature extractor:

```
python save_feature_extractor.py {vgg16, vgg19} $PATH_TO_THE_OUTPUT_FEATURE_EXTRACTOR --layers 3 4 5
```

Then run your model:

```
python eval.py inspirational_generation -n $modelName -m $modelType --inputImage $pathTotheInputImage -f $PATH_TO_THE_OUTPUT_FEATURE_EXTRACTOR
```

### I have generated my metrics. How can i plot them on visdom ?

Just run
```
python eval.py metric_plot  -n $modelName
```

## LICENSE

This project is under BSD-3 license.
