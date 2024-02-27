# Case Study and Implementation

Case Study Summary: Health Data Classification


# The Dataset
**Short Description of the Data:**

The dataset has a shape of (70000, 12) whichh is 70000 rows and 12 columns. The columns include age, gender, height,	weight,	ap_hi,	ap_lo,	cholesterol,	gluc,	smoke,	alco,	active and cardio. Each column is important in the training of the model. The task is to develop a model that can learn and generalize from this data to accurately tell status of a patient(positive or negative).

**Data Source:** The data is from Kaggle and can be found [here](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset)

I will be using the following columns to train the model:

    age: The age of the patient in days. This is then converted to years.
    gender: The gender of the patient. 1 - Female, 2 - Male
    height: The height of the patient in cm
    weight: The weight of the patient in kg
    ap_hi: Systolic blood pressure
    ap_lo: Diastolic blood pressure
    cholesterol: The cholesterol level of the patient
    gluc: The glucose level of the patient
    smoke: Whether the patient smokes or not
    alco: Whether the patient drinks alcohol or not
    active: Whether the patient is active or not
    cardio: The status of the patient (positive or negative)

The dataset is a classification problem and the target variable is cardio. The dataset is imbalanced and the target variable is binary. The dataset is preprocessed and the target variable is encoded. The dataset is split into training testing and validation sets. The model is trained on the training data and evaluated on the test data. The model is then used to make predictions on the validation data. The predictions are evaluated using the classification report and confusion matrix. The model is optimized using various techniques and the optimized model is evaluated on the test data. 

### Non optimized Model Architecture
The model architecture is a Sequential model which is a linear stack of layers. The model has 5 Dense layers and 1 Input layer and output layer. The model is compiled with the Adam optimizer and binary crossentropy loss function. The model is trained for 50 epochs with a batch size of 128. The model is evaluated on the test data and the accuracy is calculated.

Layers:

    Input Dense Layer (500 Neurons) with L2 Regularization. It has 500 neurons & applies a ReLU (Rectified Linear Unit) activation function.
    Dense Layer with (128 Neurons) & applies the ReLU activation function.
    Dense Layer with (64 Neurons) & applies the ReLU activation function.
    Dense Layer (32 Neurons) with L2 Regularization & ReLU activation function for non-linearity.
    Dense Layer (16 Neurons) with L2 Regularization & ReLU activation function for non-linearity.
    Dense Layer (8 Neurons) with L2 Regularization & ReLU activation function for non-linearity.
    Output Dense Layer (1 Neuron) with L2 Regularization & sigmoid activation function for binary classification.

### Optimized Model Architecture
The model architecture is a Sequential model which is a linear stack of layers. The model has 5 Dense layers and 1 Input layer and output layer. The model is compiled with the Adam optimizer and binary crossentropy loss function. The model is trained for 50 epochs with a batch size of 128. The model is evaluated on the test data and the accuracy is calculated.

Layers:

    Input Dense Layer (5000 Neurons) with L2 Regularization. It has 500 neurons & applies a ReLU (Rectified Linear Unit) activation function.

    Dense Layer of shape (None, 128) with l1 regularization & applies the ReLU activation function.

    Dense Layer with (64 Neurons) & applies the ReLU activation function.

    Dense Layer (32 Neurons) with l2 regularization & applies the ReLU activation function

    Dense Layer (16 Neurons) with l1 regularization & applies the ReLU activation function.

    Dense Layer (8 Neurons) applies the ReLU activation function

    Output Dense Layer (1 Neuron) & sigmoid activation function


### Model Training
The model is trained with the Adam optimizer and binary crossentropy loss function. The model is trained for 50 epochs with a batch size of 128. The model is evaluated on the test data and the accuracy is calculated.

### Model Prediction
The model is used to make predictions on the test data and the predictions are evaluated using the classification report and confusion matrix.

### Model Summary
The model summary is printed and the model architecture is visualized.

Provides a thorough and detailed discussion of various optimization techniques used, including clear explanations of underlying principles and their relevance to the project. Clearly explains the parameters associated with each optimization technique and their significance in the context of the project. Provides detailed information on how parameter values were selected or tuned, with justification for the chosen settings.

## Why the Optimized Model should be Better
Using L2 regularization, the model is able to generalize better and reduce overfitting by penalizing the weights in the model. L2 is best when we have a large number of features. L1 regularization is used to reduce the number of features in the model. It is used to reduce overfitting and improve the model's performance. Our model is able to generalize better and reduce overfitting by using L2 regularization. 

## Conclusion and Recommendations
The model is able to accurately predict the status of a patient with an accuracy of 0.73 - 0.74. 
The model is able to generalize better and reduce overfitting by using L2 regularization. 