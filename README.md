# CNN-Based Anomaly Detection in Time Series Data
This project demonstrates how to build a Convolutional Neural Network (CNN) model for anomaly detection in time series data using Keras. It is implemented in Google Colab and uses a CSV dataset containing time series values. The model detects anomalies based on reconstruction errors by setting a dynamic threshold. This approach can be applied to various domains such as sensor data, finance, and network security.

## Features
- Time series data preprocessing and scaling using MinMaxScaler
- CNN model built with Keras for sequence modeling
- Anomaly detection using reconstruction error thresholding
- Precision, Recall, and F1-score calculation for model evaluation
- Visualizing predictions and detected anomalies

## Installation

  To run this project, you'll need the following dependencies:
  
  ```bash
  pip install numpy pandas keras scikit-learn matplotlib
  ```
  
  Alternatively, you can open the Google Colab Notebook directly and run the project without any additional setup.

## Dataset
  The model expects a CSV file containing a single column, value, representing the time series data. For example:
  
  ```csv
  value
  0.1
  0.15
  0.2
  ...
  ```
  
  You can load your dataset using the following command:

  ```python
  data = pd.read_csv('/content/data.csv')
  ```

## Model Architecture
The CNN model for time series anomaly detection consists of:

- 1D Convolutional layers with 32 and 64 filters
- MaxPooling layers to downsample the data
- Flatten and Dense layers for final predictions

  ```python
  model = Sequential()
  model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(look_back, 1)))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Flatten())
  model.add(Dense(50, activation='relu'))
  model.add(Dense(1))
  ```

## Training
  The model is trained on the time series data using Mean Absolute Error (MAE) as the loss function and Adam optimizer:
  
  ```python
  model.compile(optimizer='adam', loss='mean_absolute_error')
  model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.2, verbose=2)
  ```

## Anomaly Detection
  Anomalies are detected based on reconstruction errors between the predicted and actual values. A dynamic threshold is defined as the mean error plus three times the standard deviation:

  ```python
  mean_error = np.mean(reconstruction_error)
  std_error = np.std(reconstruction_error)
  threshold = mean_error + 3 * std_error
  anomalies = np.where(reconstruction_error > threshold)[0]
  ```

## Evaluation
  Precision, Recall, and F1-score are calculated to evaluate the anomaly detection performance:

  ```python
  precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
  print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
  ```

## Visualization
  The results are plotted to visualize the true data, predicted values, and detected anomalies:
  
  ```python
  plt.figure(figsize=(15, 5))
  plt.plot(scaled_values, label='True Data')
  plt.plot(np.arange(look_back, len(predictions) + look_back), predictions, label='Predicted Data')
  plt.scatter(anomalies + look_back, predictions[anomalies], color='red', label='Anomalies')
  plt.legend()
  plt.show()
  ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
