# hand-gesture-cnn

This is a convolutional neural network that has been trained to recognize hand gestures like the fist or palm and identify them. Then, it uses Selenium to assign each gesture a browser control such as going backwards or forwards. It has been trained with a Tensorflow model over 50 epochs and has a 95.7% accuracy on the validation dataset.

When you start the program, you wait until a window called Thresholded appears and then you press "s" to start recording your hand. Once you place your hand, the program recognizes it and controls the browser (Firefo) according to the respective gesture.

To utilize this, you must download geckodriver as Firefox requires it.
