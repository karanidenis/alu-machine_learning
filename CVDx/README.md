# Case Study and Implementation

Case Study Summary: Health Data Classification


# The Dataset
**Short Description of the Data:**

The dataset has a shape of (70000, 12) whichh is 70000 rows and 12 columns. The columns include age, gender,	height,	weight,	ap_hi,	ap_lo,	cholesterol,	gluc,	smoke,	alco,	active and cardio. Each column is important in the training of the model. The task is to develop a model that can learn and generalize from this data to accurately tell status of a patient(positive or negative).

### Model Architecture
The model architecture is a Sequential model which is a linear stack of layers. The model has 9 Dense layers and 1 Input layer. The model is compiled with the Adam optimizer and binary crossentropy loss function. The model is trained for 50 epochs with a batch size of 128. The model is evaluated on the test data and the accuracy is calculated.

Layers:

    Input Dense Layer (128 Neurons) with L2 Regularization. It has 128 neurons & applies a ReLU (Rectified Linear Unit) activation function.

    Dense Layer with (64 Neurons) & applies the ReLU activation function. 
                                                                    
    Dense Layer (32 Neurons) with L2 Regularization & ReLU activation function for non-linearity.

    Output Dense Layer (1 Neuron) with L2 Regularization & sigmoid activation function for binary classification.