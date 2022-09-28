# ST-NILM

This repository hosts the official implementation of the method proposed in the paper "***ST-NILM: A Wavelet Scattering-Based Architecture for Feature Extraction and Multi-Label Classification in NILM Signals***". The paper was submitted to IEEE Sensors Journal and is currently under review.

![dfml_ScatDFML4](https://user-images.githubusercontent.com/13191527/192824107-3fa1faad-591b-41b6-a72b-860394e29938.png)

---

## Dependencies

The model was implemented on Python 3.8 using Tensorflow/Keras. To install all dependencies for this project, run:

```
$ cd ScatNILM
$ pip install -r requirements.txt
```

---

## Dict structure

This implementation uses a dictionary structure to define some of the execution parameters. The fields of this dictionary are:

* `N_GRIDS`: Total positions of the grid (default = 5).
* `N_CLASS`: Total of loads on the dataset (default = 26).
* `SIGNAL_BASE_LENGTH`: Total of mapped samples on each signal cut (default = 12800, 50 electrical network cycles).
* `MARGIN_RATIO`: Size of the unmapped margins defined by a portion of the signal. (default = 0.15).
* `DATASET_PATH`: Path to the .hdf5 file containing the samples.
* `TRAIN_SIZE`: Ratio of the examples used for training (default = 0.8). (Only used if the kfold is not performed)
* `FOLDER_PATH`: Path to the folder where the model shall be stored.
* `FOLDER_DATA_PATH`: Path to the *.p files with the already processed data. Usually it's the same that FOLDER_PATH.
* `N_EPOCHS_TRAINING`: Total of epochs for training. (default = 250)
* `INITIAL_EPOCH`: Initial epoch to continue a training, only useful if a training will be continued. (default = 0).
* `TOTAL_MAX_EPOCHS`: Max of training epochs.
* `SNRdB`: Noise level on dB.

---

## Trained weights

The trained weights may be downloaded from this [link](). Download and place the `TrainedWeights` folder inside the `ScatNILM` directory to use them.

---

## Dataset Waveforms

The LIT-Dataset is a public dataset and can be downloaded on this [link](http://dainf.ct.utfpr.edu.br/~douglas/LIT_Dataset/index.html). However, only MATLAB tools are provided. In order to use the dataset with this implementation, a version on *.hdf5 can be downloaded on the following [link](https://drive.google.com/file/d/10NL9S8BYioj1U1_phCEoKX4WWRQoBuYW/view?usp=sharing). The dataset is stored on this file in the following hierarchical structure:

- `1` -> Total number of loads on each waveform
    - `i` -> Array containing all the samples for each waveform
    - `events` -> Array containing the events array. Each event array has the same length as the waveform. If a position has a 0, there is no event. If it has a 1, there is an ON event on the sample with the same index, and if it has a -1, there is an OFF event.
    - `labels` -> Array of connected loads. The connected loads for each waveform are represented by an array with the labels of the connected loads in the order of the events. So, if there is only one appliance, the array shall look like: ["A", "A"].
- `2`
    - `i`
    - `events`
    - `labels`
- `3`
    - `i`
    - `events`
    - `labels`
- `8`
    - `i`
    - `events`
    - `labels`

To use this file with this implementation, download it and place the `Synthetic_Full_iHall.hdf5` file in the `ScatNILM` directory.

---

## How to run

To train, install all dependencies, configure the dictionary structure on the file `src/main.py` and run it as follows:

```
$ cd src
$ python3 main.py
```

Also, there are a few notebooks in the folder `notebooks` for evaluation of the models and some visualization tests.

## Tests on Jetson TX1

The tests on Jetson TX1 are detailed in this [tutorial](EmbeddedSystem.md).

---

## DeepDFML-NILM

This repository is based on the project [DeepDFML-NILM: A New CNN-Based Architecture for Detection, Feature Extraction and Multi-Label Classification in NILM Signals](https://github.com/LucasNolasco/DeepDFML-NILM). <!-- We propose a new CNN architecture to perform detection, feature extraction, and multi-label classification of loads, in non-intrusive load monitoring (NILM) approaches, with a single model for high-frequency signals. This model follows the idea of YOLO network, which provides the detection and multi-label classification of images. The obtained results are equivalent or superior (in most analyzed cases) to state-of-the-art methods for the evaluated datasets. -->

## Cite

If this work helped you somehow, here is a way to cite it:

```
@ARTICLE{Nolasco2022,
  author={Nolasco, Lucas da Silva and Lazzaretti, André Eugenio and Mulinari, Bruna Machado},
  journal={IEEE Sensors Journal}, 
  title={DeepDFML-NILM: A New CNN-Based Architecture for Detection, Feature Extraction and Multi-Label Classification in NILM Signals}, 
  year={2022},
  volume={22},
  number={1},
  pages={501-509},
  doi={10.1109/JSEN.2021.3127322}}
```
