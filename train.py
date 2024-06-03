import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Load data
with open('./data.pickle', 'rb') as file:
    data_dict = pickle.load(file)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier(random_state=42)
print("Training the model...")
model.fit(x_train, y_train)

# Predict and evaluate the model
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')
f1 = f1_score(y_test, y_predict, average='weighted')

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')

# Save the model
with open('model.p', 'wb') as file:
    pickle.dump({'model': model}, file)
print("Model saved successfully.")
