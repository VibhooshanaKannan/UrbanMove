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
file_path = r"C:\Users\vibho\OneDrive\Documents\Desktop\TRAFFIC MANAGEMENT\traffic_data.csv"  # Change this if needed
df = pd.read_csv(file_path)
# Check missing values
df.fillna(method='ffill', inplace=True)
# Encode categorical columns
label_enc = LabelEncoder()
df['traffic_density'] = label_enc.fit_transform(df['traffic_density'])  # Low, Medium, High -> 0,1,2
df['road_type'] = label_enc.fit_transform(df['road_type'])  # Encode Road Types
df['signal_status'] = label_enc.fit_transform(df['signal_status'])  # Encode Signal Colors

# Scale numerical data for better ML performance
scaler = StandardScaler()
df[['vehicle_count', 'signal_duration', 'average_speed']] = scaler.fit_transform(df[['vehicle_count', 'signal_duration', 'average_speed']])

# Save processed data
df.to_csv("/mnt/data/processed_traffic_data.csv", index=False)

print("✅ Data Preprocessing Completed")
