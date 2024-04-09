import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from category_encoders import LeaveOneOutEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf

df = pd.read_csv('data_science/data_lake/Mental_Health_Dataset.csv')


# print(f"Number of Duplicate rows :", df.duplicated().sum())
df = df.drop_duplicates()

# print(f"Number of null rows :", df.isnull().sum())

df.dropna(inplace=True)

# print(f"Number of null rows :", df.isnull().sum())

le =LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["self_employed"] = le.fit_transform(df["self_employed"])
df["family_history"] = le.fit_transform(df["family_history"])
df["treatment"] = le.fit_transform(df["treatment"])
df["Coping_Struggles"] = le.fit_transform(df["Coping_Struggles"])

data = pd.get_dummies(data=df, columns=["Occupation", "Days_Indoors", "Growing_Stress",
            "Changes_Habits", "Mental_Health_History", "Work_Interest", "Social_Weakness",
            "mental_health_interview", "care_options"])

data = pd.get_dummies(data=data, columns=["Mood_Swings"])

data.drop("Timestamp", axis=1, inplace=True)

leave_encoder = LeaveOneOutEncoder()
data["Country"] = leave_encoder.fit_transform(data["Country"], data.iloc[:, -3])

print(data.head())

y = data.iloc[:, -3:]
X = data.drop(data.iloc[:, -3:], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
print(y.head())
print(X.head())

# Define the logistic regression model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid')  # Adjust the output layer to match the number of target variables
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: ', accuracy)