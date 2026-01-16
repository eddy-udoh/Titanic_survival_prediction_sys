import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load Data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. Preprocessing 
# Fill missing Age with median
df['Age'] = df['Age'].fillna(df['Age'].median())
# Convert Sex: male=0, female=1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
# Select only the features we will use in the web app
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features]
y = df['Survived']

# 3. Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Build Neural Network (Binary Classification)
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(6,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # Sigmoid forces output between 0 and 1

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train
print("Training Titanic Model...")
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# 6. Save
model.save('titanic_model.h5')
joblib.dump(scaler, 'titanic_scaler.pkl')
print("Saved titanic_model.h5 and titanic_scaler.pkl")