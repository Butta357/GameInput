import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import numpy as np

# Load latest session
files = [f for f in os.listdir('data') if f.startswith('session_')]
if not files:
    print("No sessions found")
    exit()

latest = max(files)
df = pd.read_csv(f'data/{latest}')
print(f"Analyzing {latest}")
print(df.head())

# Plot axes
for col in df.columns[1:]:
    if 'axis' in col:
        plt.figure(figsize=(10, 6))
        plt.plot(df['timestamp'] - df['timestamp'].min(), df[col])
        plt.title(f'{col} over time')
        plt.xlabel('Time (s)')
        plt.ylabel('Axis value')
        plt.savefig(f'data/{col}_plot.png')
        plt.close()

# For personalized curve, perhaps fit a response curve
# Assuming axis 0 is main stick
if 'axis_0' in df.columns:
    x = df['axis_0']
    y = df['timestamp']  # Or something, but time is not input
    # Actually, to create input curve, perhaps the curve is how input maps to output, but since no output, maybe deadzone or sensitivity
    # For simplicity, plot histogram of axis values
    plt.figure()
    plt.hist(df['axis_0'], bins=50)
    plt.title('Axis 0 value distribution')
    plt.savefig('data/axis_0_hist.png')
    plt.close()

print("Analysis complete. Plots saved in data/")