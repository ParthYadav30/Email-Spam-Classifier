# Spam Email Classifier using MLP and TF-IDF

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers

# Load the dataset (use a CSV file that contains 'text' and 'label' columns)
# Ensure the dataset has been preprocessed so that 'target' is 1 for spam and 0 for not spam
data = pd.read_csv('/content/spam_assassin.csv')

# Preview the dataset
print(data.head())

# Check for missing values and remove them if any
data.dropna(inplace=True)

# Split into features (X) and target (y)
X = data['text']
y = data['target']

# Text preprocessing and vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X).toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the MLP model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Predict on test data
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Classification report for detailed evaluation
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

# Save the model (optional)
# model.save('spam_email_mlp_model.h5')
