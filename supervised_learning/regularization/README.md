### regularization

Regularization is the process of adding information in order to solve an ill-posed problem or to prevent overfitting.

### Regularization techniques

- L1 regularization - absolute value of the magnitude of the coefficients multiplied by a constant `alpha` (the regularization parameter). It is used to create sparse models, that is, models with few coefficients; Some coefficients can become zero and eliminated. It is used to prevent overfitting.
- L2 regularization - squared magnitude of the coefficients multiplied by a constant `alpha`. It is used to prevent overfitting. It is the most common type of regularization.

- Elastic net regularization - a linear combination of L1 and L2 regularization. It is used to prevent overfitting. It is used when there are multiple features that are correlated with one another.

- Dropout - randomly sets a fraction `rate` of input units to 0 at each update during training time, which helps prevent overfitting. The units that are kept are scaled by `1 / (1 - rate)`, so that their sum is unchanged at training time and inference time.

- Early stopping - stop training when a monitored quantity (e.g. validation loss) has stopped improving. It is used to prevent overfitting. It is useful when training for a long time.

- Data augmentation - increase the size of the training set by adding transformed copies of already existing data or artificially created data from existing data. It is used to prevent overfitting. It is useful when there is not enough data to train the model.

- Batch normalization - normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1. It is used to prevent overfitting. It is useful when training for a long time.

- Gradient noise injection - add noise to gradient during training. It is used to prevent overfitting. It is useful when training for a long time.
- ...

...
0x05-regularization/0-weights.py: def l2_reg_cost(cost, lambtha, weights, L, m): # Path: supervised_learning/regularization/0-weights.py
    def l2_reg_cost(cost, lambtha, weights, L, m): 
        """Calculates the cost of a neural network with L2 regularization
        using tensorflow
        """
        return cost + tf.losses.get_regularization_losses()
        

0x05-regularization/0-weights.py: def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L): # Path: supervised_learning/regularization/0-weights.py