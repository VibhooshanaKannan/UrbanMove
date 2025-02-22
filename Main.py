import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
import network as nx

# Load dataset
df = pd.read_csv(r"C:\Users\vibho\OneDrive\Documents\Desktop\TRAFFIC MANAGEMENT\traffic_data.csv")

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Convert timestamp to datetime and extract features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.weekday

# Encode categorical variables
categorical_cols = ['traffic_density', 'road_type', 'weather_conditions', 'current_signal_status', 
                    'route_suggestions', 'emergency_vehicle_detected', 'priority_level']

# Use get_dummies with drop_first=False to retain all categories
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Drop unnecessary columns
df.drop(columns=['timestamp', 'user_id'], inplace=True)

# Print column names to verify correct encoding
print("Columns after encoding:\n", df.columns)

# Selecting relevant features
features = ['vehicle_count', 'average_speed', 'signal_duration', 'adaptive_timing', 'hour', 'weekday'] + \
           [col for col in df.columns if 'traffic_density_' in col or 'road_type_' in col]

# Check available traffic_density columns
traffic_density_cols = [col for col in df.columns if 'traffic_density_' in col]
print("Available traffic density columns:", traffic_density_cols)

# Ensure the correct target column exists
target = 'traffic_density_High'
if target not in df.columns:
    raise KeyError(f"Column '{target}' not found. Available options: {traffic_density_cols}")

# Split data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Print final dataset info
print("Final dataset shapes:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
