# Lab 4: Recurrent Neural Network for Stock Price Prediction

**Course:** Neural Networks and Their Applications  
**Student:** Bichurina S.P., Group 09-261, 3rd year  
**Variant:** Amazon Stock Price Prediction  
**University:** Kazan (Volga Region) Federal University, Institute of Computational Mathematics and Information Technologies, Department of Information Systems

---

## Task Description

Build a recurrent neural network (RNN) using PyTorch and train it on a time series dataset. The task was to predict stock prices using historical closing price data.

## Mathematical Foundations

1. **Time sequence formation** — Input data is transformed into sliding windows of length `seq_len`, where each window contains consecutive normalized price values and the target is the next value.
2. **MinMaxScaler normalization** — Data is scaled to the [0, 1] range using the formula: x_norm = (x - x_min) / (x_max - x_min).
3. **SimpleRNN** — The RNN cell updates the hidden state h_t using the current input and the previous hidden state through learned weight matrices and bias terms. The prediction is produced via a fully connected output layer.
4. **MSE Loss** — Mean Squared Error, used as the training loss function.
5. **Inverse normalization** — Predicted values are transformed back to the original price scale for evaluation.
6. **MAE** — Mean Absolute Error, used to measure prediction quality.

## Dataset

The dataset contained 1,763 records of Amazon stock closing prices from 2010 to 2017. Data from 2010–2016 (85%) was used for training, and data from 2016–2017 was used for testing.

## Network Architecture

- **Model:** Sequential with a SimpleRNN layer (50 units, ReLU activation) followed by a Dense output layer (1 unit)
- **Sequence length:** 20
- **Optimizer:** Adam
- **Epochs:** 50
- **Batch size:** 32

## Results

On the training set, the model closely matched the actual price dynamics. On the test set (2016–2017), the following observations were made:

- The model captures the general trend of price movements, adequately reflecting periods of both decline and increase.
- Predictions tend to be smoothed relative to sharp changes in the real data — the network shows insufficient sensitivity to rapid price surges.
- **MAE on the test set: 3.6**, representing the average deviation of predicted values from actual prices.

The loss function decreased consistently over the training epochs, indicating stable learning.

## Conclusion

The RNN demonstrated good results for Amazon stock price prediction despite some limitations. While the model fits the training data well and captures overall price dynamics on the test set, it struggles with sharp price movements. For improved prediction accuracy, more advanced architectures (e.g., LSTM or GRU) and an expanded dataset could be used.
