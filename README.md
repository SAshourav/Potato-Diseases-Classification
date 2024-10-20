# Potato Plant Disease Detection

This project focuses on detecting diseases in potato plants using deep learning and image classification techniques. The goal is to help farmers identify plant diseases early by taking a picture of the potato plant and using a mobile app to determine if it has a disease or not.

## Technology Stack

### 1. **Model Building**
- **TensorFlow**: Used for building and training the CNN model.
- **CNN (Convolutional Neural Networks)**: The primary deep learning architecture used for image classification.
- **Data Augmentation**: To enhance the training dataset.
- **TF Dataset**: Efficient data loading and processing pipeline.

### 2. **Backend Server and ML Ops**
- **TensorFlow Serving**: For serving the trained model in a production environment.
- **FastAPI**: To create the backend API for image prediction.

### 3. **Model Optimization**
- **Quantization**: Optimize the model for better performance on mobile devices.
- **TensorFlow Lite**: Deploy the model on mobile devices.

### 4. **Frontend**
- **React JS**: For the web frontend.
- **React Native**: To build the mobile app for farmers to take pictures and get predictions.

### 5. **Deployment**
- **Google Cloud Platform (GCP)**: For cloud deployment and scalability.
- **Google Cloud Functions (GCF)**: Serverless functions to handle requests.

## Model Training

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Model Building
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # Assuming 3 classes: Early Blight, Late Blight, Healthy
])

# Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
history = model.fit(train_ds, validation_data=val_ds, epochs=10)
