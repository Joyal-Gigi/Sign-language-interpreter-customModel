# Sign-language-interpreter-customModel
American sign language interpreter made using random Forest classifier. 
(Data used for creation of dataset is not uploaded. However it can be created using the datacollection code.)




Random Forest Algorithm
Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.
As the name suggests, "Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset." Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.
The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.
Ensemble learning, in general, is a model that makes predictions based on a number of different models. By combining individual models, the ensemble model tends to be more flexible🤸‍♀️ (less bias) and less data-sensitive🧘‍♀️ (less variance).


CODE explanation
						
      #dataset creation

Configures mediapipe for hand tracking.

Initializes the hands module.

Defines the directory path for the data.

Initializes empty lists for storing data and labels.

Iterates over directories in the data directory.

Iterates over image paths in each directory.

Initializes temporary variables for data and coordinates.

Reads an image and converts it to RGB.

Processes the image with the hands module to detect hand landmarks.

If hand landmarks are detected:

Iterates over detected hand landmarks.

Extracts the x and y attributes of the landmark points.

Calculates the differences between x and y attributes and their respective minimum values.

Appends the calculated data to the temporary data list.

Appends the temporary data and the directory name (label) to the data and labels lists.

Creates a pickle dataset file named 'data.pickle' and dumps the data and labels into it.


					#testing and training

Imports necessary libraries: pickle for loading and saving data, RandomForestClassifier from scikit-learn for classification, train_test_split from scikit-learn for splitting data into training and testing sets, accuracy_score from scikit-learn for calculating accuracy, and numpy for array operations.

Loads the data dictionary from the pickle file generated in the previous code.

Converts the data and labels to NumPy arrays.

Splits the data into training and testing sets using train_test_split. 80% of the data is used for training and 20% for testing. The splitting is done randomly (shuffle=True) and ensures that the class distribution is maintained (stratify=labels).

Initializes a RandomForestClassifier.

Trains the classifier using the training data (x_train and y_train).

Tests the classifier by predicting the labels for the testing data (x_test).

Calculates the accuracy of the classifier by comparing the predicted labels with the true labels (y_test) using accuracy_score.

Prints the accuracy score.

Saves the trained model as a dictionary in a file named 'model.p' using pickle.




					#detection



Loads the trained model from the pickle file.

Initializes the video capture from the default camera.

Configures mediapipe for hand tracking.

Initializes the hands module.

Maps labels to letters for character prediction.

Enters a loop to process each frame from the video capture.

Processes the frame with mediapipe to obtain hand landmarks.

If hand landmarks are detected, draws them on the frame.

Extracts the x and y coordinates of the hand landmarks and calculates differences from the minimum values.

Calculates the bounding box coordinates based on the extracted landmarks.

Makes a prediction using the trained model by passing the extracted data as input.

Retrieves the predicted character based on the label.

Draws a rectangle around the hand region and displays the predicted character on the frame.

Displays the frame with the added annotations.

Waits for a key press and repeats the process for the next frame.

Releases the video capture and closes all windows when the loop is terminated.
