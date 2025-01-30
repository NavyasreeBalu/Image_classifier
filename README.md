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

### Clone the repository

```bash
git clone https://github.com/your-username/AI-Programming-Python.git
cd AI-Programming-Python
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Make sure to have Python 3.6+ installed.

## Usage

To use the image classifier, run the following command:

```bash
python classifier.py --image_path path_to_image --checkpoint checkpoint.pth --category_names category_names.json --gpu
```

Where:
- `--image_path` is the path to the image you want to classify.
- `--checkpoint` is the path to the saved model checkpoint.
- `--category_names` is an optional JSON file mapping category indices to category names.
- `--gpu` (optional) uses GPU for computation if available.

## Project Structure

```
AI-Programming-Python/
│
├── Final_image_classifier_project.py    # Main script for classifying images.
├── model.py                # Defines the neural network model.
├── train.py                # Script for training the model.
├── predict.py              # Script for making predictions.
├── utils.py                # Utility functions for preprocessing and data loading.
├── requirements.txt        # List of dependencies.
├── checkpoint.pth          # Pretrained model checkpoint.
└── cat_to_name_.json     # Mapping of category indices to names.
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---