# AI Programming with Python - Image Classifier Project

This project is part of Udacity's AI Programming with Python Nanodegree program. In this project, I developed an image classifier using PyTorch and later transformed it into a command-line application.

## Project Overview

In this project, I:
- Built an image classifier using the PyTorch library.
- Trained the model using a dataset of images.
- Created a command-line interface (CLI) to make predictions on new images.

## Key Features

- **Image Classification**: Classifies images from a set of categories.
- **PyTorch Implementation**: Utilized PyTorch for deep learning model development and training.
- **Command-Line Interface (CLI)**: Convert the image classifier into a command-line application, allowing users to classify images using simple commands.

## Installation

To get started with the project locally, clone this repository and install the dependencies.

Make sure to have Python 3.6+ installed.

The imports includes:
- `torch` (PyTorch) for model development and training.
- `torchvision` for datasets and pretrained models.
- `matplotlib` for displaying images.
- `Pillow` (PIL) for image processing.
- `json` for handling category names mapping.

## Usage

To use the image classifier, run the following command:

### 1. Training the Model:

```bash
python python train.py flowers/ --arch resnet34 --learning_rate 0.001 --hidden_units 512 --epochs 10 --gpu
```

Where:
- `flowers/` is the path to the image dataset (including train, test, and validation subdirectories).
- `--arch` specifies the architecture of the model (e.g., `vgg16`, `resnet34`).
- `--learning_rate` sets the learning rate (e.g., `0.001`).
- `--epochs` specifies the number of epochs for training (e.g., `10`).
- `--gpu` uses GPU for training if available.

### 2. Classifying an Image:

Once the model is trained, you can classify a new image using the following command:

```bash
python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --category_names cat_to_name.json --top_k 3 --gpu
```

Where:
- `flowers/test/1/image_06743.jpg` is the path to the image you want to classify.
- `checkpoint.pth` is the saved model checkpoint file.
- `--category_names` is a JSON file that maps category indices to names (e.g., `cat_to_name.json`).
- `--top_k` specifies the number of top predictions to show (default is `1`).
- `--gpu` uses GPU for computation if available.

### Expected Output:

```text
Top 1 - Category: Daisy, Probability: 0.85
Top 2 - Category: Rose, Probability: 0.10
Top 3 - Category: Tulip, Probability: 0.05
```

## Project Structure

```
AI-Programming-Python/
│
├── Image_Classifier_Project.py    # Main script for classifying images.
├── train.py                # Script for training the model.
├── predict.py              # Script for making predictions.
├── checkpoint.pth          # Pretrained model checkpoint.
└── cat_to_name.json        # Mapping of category indices to names.
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--S
