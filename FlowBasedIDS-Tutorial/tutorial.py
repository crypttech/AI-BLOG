import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Dataset loading..
X_train = np.load("train_data.npy")
y_train = np.load("train_label.npy")
X_test = np.load("test_data.npy")
y_test = np.load("test_label.npy")


# Validation set split parameters
validation_rate = 0.3
validation_size = int(len(y_test)*0.3)

# Model hyperparameter configuration
batch_size = 100
epochs = 3
learning_rate = 0.0005

# Scaling(Zero mean and unit stderr) Step
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Simple Neural Network Model Creating...
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units=40, kernel_initializer='uniform', activation='relu', input_dim=75))
# Adding the second hidden layer
classifier.add(Dense(units=20, kernel_initializer='uniform', activation='relu'))
# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Defining optimization function.
optimizer = optimizers.Adam(lr=learning_rate)

# Format the name of model weights.
weight_name = 'weights.{epoch:02d}-{acc:.4f}.h5'

# If improve results save model weights like checkpoint approach
checkpoint = ModelCheckpoint(weight_name, save_best_only=True)

# If results not improved after sequential 3 epoch just stop early the training.
stopper = EarlyStopping(monitor='val_loss', patience=3, mode='auto')

# Compile classifier model with hyperparameters
classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

# To finetune model you should load existing model weights.
# classifier.load_weights("weights.h5")

# Start training with determined hyperparameters.
classifier.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint, stopper],
               validation_data=[X_test[:validation_size], y_test[:validation_size]])

# Test the model with test data set.
y_pred = classifier.predict(X_test[validation_size:])

# Thresholding results
y_pred = (y_pred > 0.5)

# Calculate and print the confusion matrix of results
cm = confusion_matrix(y_test[validation_size:], y_pred)
print("\nConfusion Matrix \n", cm)
