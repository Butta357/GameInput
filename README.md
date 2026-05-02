# Xbox Controller Input Monitor

This Python application monitors input from an Xbox controller, records sessions to CSV files, and provides analysis tools to visualize and understand input patterns for creating personalized input curves.

## Features

- Real-time monitoring of controller axes, buttons, and hats
- Session recording with timestamps
- Data export to CSV
- Analysis scripts for visualization

## Requirements

- Python 3.x
- Xbox controller connected

## Installation

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Connect your Xbox controller.
2. Run the application:
   ```
   python main.py
   ```
   This opens a GUI window with buttons to start/stop recording and analyze data.

3. Click "Start Recording" to begin capturing input data to a CSV file in the `data/` directory.
4. Click "Stop Recording" to end the session.
5. Click "Analyze Latest" to process the most recent session and generate plots.

## Analysis

The analyzer creates plots of axis values over time and histograms to help identify input patterns, deadzones, and sensitivity for personalizing controller settings.

For more advanced curve fitting, modify `analyzer.py` as needed.