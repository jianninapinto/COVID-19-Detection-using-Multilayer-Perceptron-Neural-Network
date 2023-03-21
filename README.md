
# COVID-19 Detection using Multilayer Perceptron Neural Network

The aim of this project is to develop a neural network that detects COVID-19 in lung scans. The Multilayer Perceptron (MLP) neural network is used to train a model using a publicly available SARS-CoV-2 CT scan dataset containing 1252 CT scans positive for SARS-CoV-2 infection (COVID-19) and 1229 CT scans for non-infected patients. The dataset has been collected from real patients in hospitals in Sao Paulo, Brazil.

## Dataset

The dataset can be downloaded from Kaggle at the following link: https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset. The dataset contains CT scans of the chest for patients with COVID-19 and without COVID-19. Each scan has been labeled as COVID-19 positive or negative. The dataset is split into two folders, one for positive scans and one for negative scans.

## Train-Validation-Test Split

The dataset is divided into a train set (70%), a validation set (20%), and a test set (10%). The training set is used to train the neural network, the validation set is used to tune the hyperparameters, and the test set is used to evaluate the final model.


## Neural Network Architecture

The MLP neural network architecture used for this task consists of the following layers:
- Input layer (Rescaling layer)
- Flatten layer (to convert 2D image data to a 1D vector)
- Hidden layer 1 (32 neurons, sigmoid activation)
- Hidden layer 2 (32 neurons, sigmoid activation)
- Output layer (1 neuron, sigmoid activation)

Different variations of the neural network architecture were tested by adding more hidden layers and varying the activation functions and number of neurons.

## Training

The neural network was trained using the Adam optimizer and binary cross-entropy loss. The hyperparameters used for training the neural network were tuned on the validation set. To tune the hyperparameters of the model, we experimented with different combinations of neurons and activation functions (ReLu and sigmoid). Also, different regularization techniques, including Early Stopping, L1 regularization, L2 regularization, and Dropout Regularization were tested.

## Results

The results of our experiments are summarized in the table below:

| Number of Units | Activation Functions | Regularization | Validation Accuracy | Validation Loss |
| --- | --- | --- | --- | --- |
| 32 | sigmoid | None | 69.49% | 0.6459 |
| 64 | sigmoid | None | 70.30% | 0.6223 |
| 32 | ReLu    | None | 71.52% | 0.5552 |
| 64 | ReLu    | None | 71.52% | 0.5347 |
| 64 | ReLu    | EarlyStopping | 71.92% | 0.5288 |
| 64 | ReLu    | EarlyStopping + L1 | 72.12% | 0.5374 |
| 64 | ReLu    | EarlyStopping + L2 | 70.71% | 0.6630 |
| 64 | ReLu    | EarlyStopping + Dropout | 71.72% | 0.6728 |


## Conclusion

In conclusion, the MLP neural network is trained to detect COVID-19 in lung scans. This model could be used as a tool for detecting COVID-19 in lung scans, and it has the potential to assist radiologists in their diagnosis. Further  validation and evaluation is needed before the model can be used in a clinical setting.

