Human Activity Recognition (HAR) Using Pretrained Models

This project implements a Human Activity Recognition (HAR) system using machine learning and deep learning techniques, leveraging pretrained models. The focus is on creating an efficient pipeline for data processing, model training, evaluation, and deployment.

Table of Contents-

Introduction-

Project Workflow

Setup Instructions

Model Training and Hugging Face Integration

Issues and Troubleshooting

References

 # Introduction #

Human Activity Recognition (HAR) aims to classify various physical activities performed by humans based on sensor data. This project explores HAR using pretrained models, focusing on:

Data preprocessing

Training and fine-tuning models

Leveraging Hugging Face models for transfer learning

Evaluating and visualizing results

Project Workflow

Step 1: Data Preparation

Load Dataset: Import and inspect the dataset containing sensor data.

Data Cleaning:

Remove duplicates.

Handle missing values (NaN/null).

Normalize sensor readings.

Feature Engineering: Extract meaningful features from raw data.

Split Dataset: Divide the dataset into training, validation, and test sets.

Step 2: Model Selection and Training

Pretrained Model Usage: Use pretrained models from Hugging Face (e.g., BERT or a time-series specific model).

Fine-tuning: Adjust the pretrained model to adapt to HAR-specific tasks.

Training:

Define loss function and optimizer.

Train the model on the training dataset.

Evaluate performance on the validation set.

Step 3: Model Evaluation

Compute metrics such as accuracy, precision, recall, and F1-score.

Visualize results through confusion matrices and performance graphs.

Step 4: Deployment

Export the model for production.

Create a simple interface for predictions (e.g., using Streamlit or Flask).

Setup Instructions

Clone Repository:

git clone <repository_url>
cd <repository_directory>

Install Dependencies:

pip install -r requirements.txt

Prepare Dataset:

Place your dataset in the data/ directory.

Ensure proper formatting as per the script requirements.

Run the Notebook:
Open the Jupyter Notebook and execute cells step-by-step:

jupyter notebook HAR_Project_usingpretrandmodel.ipynb

Model Training and Hugging Face Integration

Hugging Face Models:

Import pretrained models using the transformers library.

Example:

from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("<model_name>")
tokenizer = AutoTokenizer.from_pretrained("<model_name>")

Fine-tuning:

Adjust the pretrained model for HAR tasks by freezing certain layers and modifying output layers.

Save and Load Models:

Save the trained model:

model.save_pretrained("model_output/")

Reload for inference:

from transformers import AutoModel
model = AutoModel.from_pretrained("model_output/")

Issues and Troubleshooting

Dataset Issues:

Problem: Missing or inconsistent data.

Solution: Use pandas for cleaning and filling missing values.

Model Training:

Problem: Slow training speed.

Solution: Optimize batch size and use GPU acceleration.

Hugging Face Integration:

Problem: Version mismatch.

Solution: Ensure transformers library is up-to-date:

pip install --upgrade transformers

Deployment Issues:

Problem: Model size too large.

Solution: Use model quantization techniques to reduce size.

References

Hugging Face Transformers Documentation

Pandas Documentation

NumPy Documentation

This README provides a comprehensive guide to understanding, implementing, and troubleshooting the HAR project. Follow the steps systematically to ensure a smooth workflow and successful deployment.

