import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv('Blockbuster_Dataset_encoded.csv')

# Remove duplicates
df = df.drop_duplicates()

# Split the dataset into features and target variable
y = df['Cycle time ']
x = df.drop('Cycle time ', axis=1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

# Initialize the RandomForest model
model = RandomForestRegressor(max_depth=4, random_state=35)

# Train the model
model.fit(x_train, y_train)

# Save the trained model to a .pkl file
joblib.dump(model, 'model.pkl')

print("Model has been saved as 'model.pkl'")
