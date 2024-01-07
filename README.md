# Simple Linear Regression Neural Network


<div align="center">
  <br>
      <img src="https://github.com/RJohnPaul/Linear_regression_model/blob/b41c8beedefb75ccb431dce3bb1c2618f67051c6/Frame%2011.png" alt="Project Banner">
  </br>
</div>

<div align="center">
  <br>
      <img src="https://github.com/RJohnPaul/Linear_regression_model/blob/e25c11aa7a88815fb74a4ff01ad8959eb4ba883a/Frame-5.png" alt="Project Banner">
  </br>
</div>

</br>

My Very First Model Trained . This Python script demonstrates the creation and training of a simple neural network for linear regression using TensorFlow and NumPy.

## Usage

Run the script using a Python interpreter. Upon execution, the script will prompt the user to train the neural network. If the user chooses to train the model (`y`), the script will fit the neural network to the given data. Otherwise, it will print a warning that the neural network is not trained.

```bash
python simple_linear_regression_nn.py
```

## Neural Network Architecture

The neural network consists of one layer with one neuron, making it a simple linear regression model. The model is compiled with Stochastic Gradient Descent (`sgd`) as the optimizer and Mean Squared Error (`mean_squared_error`) as the loss function.

```python
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
```

## Input Data

The input data (`xs`) and corresponding output data (`ys`) are provided for training the neural network. The script uses NumPy to create arrays for input and output data.

```python
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
```

## Training the Neural Network

The user is prompted to decide whether to train the neural network. If training is requested, the model is fitted to the input and output data using the `model.fit` method.

```python
user_request = input("Do you want to train the neural network? (y/n) ")
if user_request.lower() == "y":
    model.fit(xs, ys, epochs=500)
    print("Training complete.")
else:
    print("Training skipped. (Beware: The neural network is not trained.)")
```

## Making Predictions

After training or if the user chooses to skip training, the script makes a prediction using the trained model. In this example, the script predicts the output for the input value `10.0`.

```python
print(model.predict([10.0]))
```

## Modules Used

The script utilizes the following Python modules:

- [TensorFlow](https://www.tensorflow.org/) - An open-source machine learning framework.
- [NumPy](https://numpy.org/) - A powerful library for numerical operations in Python.

Ensure these modules are installed in your Python environment before running the script.

```bash
pip install tensorflow numpy
```

<div align="center">
  <br>
      <img src="https://github.com/RJohnPaul/SVG-Clock-with-NiceGUI/blob/17fe67997a37c39514287d6d91f6b6641ad1bbe1/Frame%209.png" alt="Project Banner">
  </br>
</div>

---
