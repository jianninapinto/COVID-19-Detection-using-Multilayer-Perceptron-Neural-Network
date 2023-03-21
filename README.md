
# COVID-19 Detection using Multilayer Perceptron Neural Network

The aim of this project is to develop a neural network that detects COVID-19 in lung scans. The Multilayer Perceptron (MLP) neural network is used to train a model using a publicly available SARS-CoV-2 CT scan dataset containing 1252 CT scans positive for SARS-CoV-2 infection (COVID-19) and 1229 CT scans for non-infected patients. The dataset has been collected from real patients in hospitals in Sao Paulo, Brazil.

## Dataset

The dataset can be downloaded from Kaggle at the following link: https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset. The dataset contains CT scans of the chest for patients with COVID-19 and without COVID-19. Each scan has been labeled as COVID-19 positive or negative. The dataset is split into two folders, one for positive scans and one for negative scans.
## Train-Validation-Test Split

The dataset is divided into a train set (70%), a validation set (20%), and a test set (10%). The training set is used to train the neural network, the validation set is used to tune the hyperparameters, and the test set is used to evaluate the final model.


## Neural Network Architecture

The MLP neural network architecture used for this task consists of the following layers:
- Input layer (Rescaling layer)
- Hidden layer 1 (256 neurons, ReLU activation)
- Hidden layer 2 (128 neurons, ReLU activation)
- Hidden layer 3 (64 neurons, ReLU activation)
- Hidden layer 4 (32 neurons, ReLU activation)
- Output layer (1 neuron, sigmoid activation)

## Training

The neural network was trained using the Adam optimizer and binary cross-entropy loss. The hyperparameters used for training the neural network were tuned on the validation set. The best-performing model reached 75.15% validation accuracy and 0.4905 validation loss compared to the baseline model which reached 71.31% validation accuracy and 0.5699 validation loss.



## Evaluation
The final model is evaluated on the test set and achieved an accuracy of 76.80% and a loss of 0.5713. The loss value of 0.5713 represents the average discrepancy between the predicted probabilities of COVID-19 cases and the actual COVID-19 cases in the test set.

## Conclusion

In conclusion, the MLP neural network is trained to detect COVID-19 in lung scans. The best-performing model achieved an accuracy of 75.15% on the validation set and 76.80% on the test set. This model could be used as a tool for detecting COVID-19 in lung scans, and it has the potential to assist radiologists in their diagnosis. Further  validation and evaluation is needed before the model can be used in a clinical setting.

