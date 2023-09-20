# Deep Learning Image Classifier

This is a Deep Learning image classifier that can classify images as "Sad" or "Happy" faces. The classifier uses a convolutional neural network (CNN) implemented in TensorFlow. Below is an overview of the key components of this project:

## Data Ingestion

The `data_ingestion` class is responsible for ingesting image data from a specified directory. It performs the following tasks:
- Iterates through subdirectories representing classes (e.g., "Sad" and "Happy").
- Verifies and preprocesses images, removing any invalid or non-supported image files.
- Rescales pixel values to the range [0, 1].
- Returns the ingested data as a TensorFlow dataset.

## Train-Test Split

The `train_test_split` class takes the ingested data and splits it into training, validation, and test datasets. The split ratios can be configured as needed.

## Model Building

The `model_building` class defines a CNN model for image classification. The architecture includes convolutional layers, max-pooling layers, and fully connected layers. The model is compiled with the Adam optimizer and binary cross-entropy loss for binary classification.

## Model Training

The `model_training` class trains the model using the training data and validates it using the validation data. It also plots the training and validation loss and accuracy over time.

## Prediction

The `prediction` class takes a trained model and an image file path as input. It uses the model to predict whether the image contains a "Sad" or "Happy" face and prints the predicted class.

## FastAPI Application

The `main.py` file contains a FastAPI application that exposes an endpoint for making predictions using the trained model. You can run the FastAPI application and make predictions by sending POST requests to the endpoint.

## Airflow Pipeline

The `airflow_pipeline.py` file defines an Airflow pipeline to automate the entire process of data ingestion, model training, and prediction. You can use Airflow to schedule and monitor the execution of these tasks.

```python
# Sample code:
# In main.py, create an instance of FastAPI and define an endpoint for predictions.
# Run the FastAPI application using uvicorn or your preferred ASGI server.

# In airflow_pipeline.py, define Airflow tasks for data ingestion, model training,
# and prediction. Set up the DAG (Directed Acyclic Graph) to schedule and execute these tasks.

# Example usage:
# - Access the FastAPI endpoint to make predictions.
# - Use Airflow to automate the entire process with scheduling and monitoring.

# Note: Make sure to configure environment variables, file paths, and settings as needed in both files.
