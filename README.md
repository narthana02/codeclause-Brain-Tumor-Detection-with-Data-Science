# codeclause-Brain-Tumor-Detection-with-Data-Science


### Step 1: Data Preparation

1. **Download and Extract Data:**
   - Download the dataset from Kaggle using the provided link.
   - Extract the contents of the downloaded zip file.

2. **Install Required Libraries:**
   - Open a terminal or command prompt and run the following command to install the necessary Python libraries:

   ```
   pip install numpy pandas opencv-python scikit-learn tensorflow
   ```

3. **Import Libraries:**
   - In your Python script or Jupyter Notebook, import the required libraries:

   ```python
   import os
   import numpy as np
   import cv2
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import LabelEncoder
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
   from tensorflow.keras.utils import to_categorical
   ```

4. **Load and Preprocess Data:**
   - Set the data directory path, categories, and initialize lists to store images and labels.
   - Loop through the categories and images, loading, resizing, and normalizing the data.

5. **Split Data into Training and Testing Sets:**
   - Use `train_test_split` from scikit-learn to split the data into training and testing sets.

6. **Normalize Data:**
   - Normalize the pixel values in the image data to the range [0, 1].

### Step 2: Build and Train the Model

1. **Encode Labels:**
   - Use `LabelEncoder` to encode the categorical labels as one-hot encoded vectors.

2. **Create the Convolutional Neural Network (CNN) Model:**
   - Build a simple CNN model using TensorFlow and Keras. Customize the architecture as needed.

3. **Compile the Model:**
   - Compile the model specifying the optimizer, loss function, and metrics.

4. **Train the Model:**
   - Train the model using the training data. You can adjust the number of epochs and batch size as needed.

### Step 3: Evaluate the Model

1. **Evaluate Accuracy:**
   - After training, evaluate the model's accuracy on the testing data and print the result.

2. **Inference on New Images:**
   - Create a function to make predictions on new images. You can pass the path to a new image as input and get the model's prediction.

### Data Visualization

- Visualize the data distribution, sample images, and model architecture using libraries like Matplotlib and Seaborn.

### Performance Visualization

- Plot training and validation loss and accuracy to monitor model performance during training.

- Visualize the results of your model by plotting the original image along with the predicted class for inference.


