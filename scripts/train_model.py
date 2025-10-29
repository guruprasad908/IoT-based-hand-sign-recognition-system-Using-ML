# scripts/train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import os

# Load the dataset
dataset_path = r"C:\Users\admin\Desktop\ll\hand_sign_iot_project\dataset\dataset_landmarks\combined_dataset.csv"
df = pd.read_csv(dataset_path)

# Separate features and labels
X = df.drop('label', axis=1).values
y = df['label'].values

# Encode the labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[es], batch_size=8)

# Save the model
os.makedirs("../model", exist_ok=True)
model.save("../model/hand_sign_model.h5")

# Save label encoder classes
np.save("../model/label_classes.npy", encoder.classes_)

print("✅ Model trained and saved to model/hand_sign_model.h5")
