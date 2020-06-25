# Image Classification
The purpose of this project is to show you how you can configure, build and train a Deep Neural Network with over 90% accuracy in distinguishing an image of a cat from a non-cat in a picture. 


# Dataset
We use a small dataset split as training, validation and testing dataset, containing pixel values of images and the corresponding image labels as cat (1) and non-cat (0). Sizes of individual datasets are as follows;
- Training set: 169 images and labels respectively
- Validation set: 40 images and labels respectively
- Testing set: 50 images and labels respectively

Each image is of shape (64, 64, 3) where 3 is for the 3 colour channels, RGB, and we are interested in evaluating how well the model performs in predicting pictures it has never seen before having been trained on such a small dataset.

These datasets are contained in the `dataset` folder, saved as [`.h5`](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) files, containning multidimensional arrays of the images, and are loaded with the  function `load_data()` with the help of the `data_util.py` python script.



# Model Architecture
As the problem is about image classification, we used [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) (or convnets or CNN), which have been extensively used in the field of computer vision, for feature representation. These convnets get their names from the concept of sliding (or convolving) a small window (also known as filters or kernels) over an image (or data sample), thereby specialising in being able to capture spatial relationships between data points of each sample. 

The network comprises of 4 layers (3 hidden and 1 output) to process spatial patterns in each image, and is broken down as follows;
- `First Hidden Layer`: This layer is a convolutional layer, consisting of 8 filters of size 3x3, a [`ReLU`](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function and takes in 64x64 pixel pictures of "cats" and "non-cats". Following the convolutional layer is a layer of computational operations called [`MaxPooling2D()`](https://keras.io/api/layers/pooling_layers/max_pooling2d/#maxpooling2d-class), which is used to reduce computational complexity by downsampling the feature maps by a factor of 2. Next is another layer of computational operations called [`Dropout()`](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf), for reducing the risk of overfitting the model to the training data. 

- `Second Hidden Layer`: This is also a convolutional layer with the same attributes as the first layer, with the exception that the inputs are the activations from the first convolutional layer. The outputs (or activations) of this layer are then passed to another layer of computational operations termed [`Flatten()`](https://stackoverflow.com/questions/43237124/what-is-the-role-of-flatten-in-keras), for converting the three-dimensional activation map output of the second convolutional layer to a one-dimensional array.
 
The first convolutional layer learns to represent simple features in the image like straight lines and edges, and then the second convolutional layer recombines these simple feaatures into more abstract representations.

- `Third Hidden Layer`: To recombine these abstract representations, we add a [`dense`](https://heartbeat.fritz.ai/classification-with-tensorflow-and-dense-neural-networks-8299327a818a) layer, with 16 neurons, that map these spatial abstract features to a particular class of images (extracted features), before feeding these identified features to the output layer.

- `Output Layer`: This layer is a dense layer with a single neuron, which receives the extracted features, and with a [`sigmoid`](https://en.wikipedia.org/wiki/Sigmoid_function) function, encodes the probability that the network is looking at one class or the other. If the probability is greater or equal to 0.5, we classify the image as a cat, else we classify the image as a non-cat.

Before fitting the model to the data, the following had to be defined as part of a compilation step to get the network ready for training;
- A gradient-based optimization technique used in training neural networks, such that they update the weights of the network based on the data it sees and the loss function. For this, we chose the [`rmsprop`](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp) optimizer.
- A loss function that computes the cross-entropy between true labels and predicted labels. The model will use this function to measure its performance on the training data, and subsequently how it is able to steer itself in the right direction towards global minimum. For this, we chose [`binary_crossentropy`](https://stackoverflow.com/questions/42081257/why-binary-crossentropy-and-categorical-crossentropy-give-different-performances). 
- A metrics to calculate how often the predicted labels match the true labels. For this, we chose [`accuracy`](https://www.kdnuggets.com/2016/12/best-metric-measure-accuracy-classification-models.html).


# Model Evaluation
The goal of machine learning is to achieve a model that generalizes well on never-before-seen data, and overfitting is the central obstacle. To monitor the model's performance and evaluate if it is overfitting the training data, we hold-out a portion of the training data - called validation data. So the model is trained on the training set, and evaluated on the validation data to assess how well the model generalizes. An accuracy of 94% was observed of the model on the training data while the model achieved an accuracy of 93% on the validation.

After fine tuning and selecting the above hyper-parameters for the individual layers, and achieving the accuracy scores on the training and validation data, the model was fed the testing data. It was interesting to see that the model achieved an outstanding accuracy score of 92%, i.e. out of every 100 new images passed to the model, the model classified 92 of them correctly as "cats" or "non-cats". This is very impressive for a model trained on such a small dataset.

To further evaluate the performance of the model, 9 new images were loaded, transformed and passed to the model, and the model correctly classified every one of these images. Remarkable result!
