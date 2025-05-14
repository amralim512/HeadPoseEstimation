# Real-time Head Pose Analysis and Orientation Classification

## Objective

This project explores computer vision techniques for real-time head pose analysis and orientation classification. It is divided into two milestones:

* **Milestone 1:** Perform head pose estimation (yaw, pitch, roll) using traditional machine learning algorithms: Support Vector Machines (SVM), Gradient Descent-based Regression, Random Forest Regression, and K-Nearest Neighbors (KNN).
* **Milestone 2:** Extend the work to face orientation classification (e.g., "Left," "Right," "Up," "Down," "Front") using an Encoder-Decoder Network based on the head pose angles predicted in Milestone 1.

By completing both milestones, students will develop an end-to-end pipeline for analyzing head pose and classifying orientation from real-time video.

## Milestone 1: Head Pose Estimation Using Machine Learning

### Description

Head pose estimation, the prediction of a person's head orientation (yaw, pitch, and roll), is crucial for various applications such as human-computer interaction, driver monitoring systems, and augmented reality. This milestone focuses on a traditional machine learning approach to predict these continuous head angles from an image.

### Requirements

1.  **Data Cleaning & Preprocessing:**
    * Handle any missing or inconsistent data present in the chosen dataset.
    * Normalize or scale the input features to ensure optimal model performance.
    * Implement feature selection techniques if necessary to reduce the dimensionality of the feature space.

2.  **Feature Extraction:**
    * Detect facial landmarks using libraries like Dlib, OpenCV, or Mediapipe.
    * Extract meaningful geometric features from these landmarks, such as:
        * Distances between key pairs of landmarks.
        * Angles formed by specific sets of landmarks.
        * Optionally, explore Histogram of Oriented Gradients (HOG) features.

3.  **Machine Learning Model Training:**
    * Train four distinct regression models:
        1.  Support Vector Machines (SVM)
        2.  Gradient Descent-based Regression
        3.  Random Forest Regression
        4.  K-Nearest Neighbors (KNN)
    * Optimize the hyperparameters of each model using techniques like grid search or cross-validation to achieve the best possible performance.

4.  **Model Comparison & Evaluation:**
    * Evaluate the performance of each trained model using the following metrics for each of the yaw, pitch, and roll predictions:
        1.  Mean Absolute Error (MAE)
        2.  Root Mean Squared Error (RMSE) (as a secondary metric)
    * Provide a comprehensive analysis comparing the performance of the different models. Justify why one model might perform better than others based on the characteristics of the extracted features and the inherent properties of each algorithm.

5.  **Real-time Head Pose Estimation on Video:**
    * Implement a real-time head pose estimation system that utilizes the best-performing trained model.
    * Integrate this system with a webcam feed using OpenCV.
    * Display
  
## Milestone 2: Face Orientation Classification Using an Encoder-Decoder Network

### Description

Building upon Milestone 1, this milestone shifts the focus from predicting continuous head angles to classifying discrete face orientations. This classification approach is often more practical for real-world applications like driver attention monitoring or facial recognition systems. An Encoder-Decoder Network, a type of deep learning model, will be trained to categorize head orientation based on the head pose angles obtained in the previous milestone.

### Requirements

1.  **Convert Head Pose Angles into Categories:**
    * Define discrete categories for face orientation based on the predicted yaw and pitch angles from Milestone 1:
        1.  **Front:** Yaw $\approx$ 0°
        2.  **Left:** Yaw < -30°
        3.  **Right:** Yaw > 30°
        4.  **Up:** Pitch > 20°
        5.  **Down:** Pitch < -20°

2.  **Feature Extraction & Preprocessing:**
    * Utilize the same extracted features from Milestone 1 as input for the classification model.
    * Ensure the data is properly normalized and preprocessed as required for training a deep learning model.

3.  **Train an Encoder-Decoder CNN for Classification:**
    * Implement an Encoder-Decoder Convolutional Neural Network (CNN) using TensorFlow/Keras.
    * **Encoder:** Design a CNN-based encoder to extract relevant features from the input data.
    * **Decoder:** Implement fully connected layers in the decoder to perform the classification into the defined face orientation categories.
    * Train the Encoder-Decoder network to accurately classify the face orientation.

4.  **Model Evaluation:**
    * Evaluate the performance of the trained classification model using standard classification metrics:
        1.  Accuracy
        2.  Precision, Recall, and F1-score for each of the defined head pose categories.
        3.  Generate and analyze a confusion matrix to understand the types of misclassifications made by the model.

5.  **Real-time Face Orientation Classification:**
    * Extend the real-time application developed in Milestone 1 to display the predicted face orientation label (e.g., "Left," "Right") as an overlay on the live video stream.

## Tools and Libraries

* **Python**
    * TensorFlow/Keras (for deep learning in Milestone 2)
    * OpenCV (for image and video processing, real-time implementation)
    * NumPy (for numerical computations)
    * Matplotlib (for plotting and visualization)
    * Pandas (for data manipulation and analysis)
    * Dlib or Mediapipe (for facial landmark detection)
* Any suitable Integrated Development Environment (IDE) such as PyCharm or Jupyter Notebook.
