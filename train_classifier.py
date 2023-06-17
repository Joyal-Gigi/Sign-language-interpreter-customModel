import pickle

from sklearn.ensemble import RandomForestClassifier  #import classifier
from sklearn.model_selection import train_test_split  #import data splitter for training and testing
from sklearn.metrics import accuracy_score  #import accuracy calculator
import numpy as np

# Load the data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets (80% training, 20% testing)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize a RandomForestClassifier
model = RandomForestClassifier()

#train classifier
model.fit(x_train, y_train)

#test classifier
y_predict = model.predict(x_test)

# Calculate the accuracy of the classifier
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model as a dictionary
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()