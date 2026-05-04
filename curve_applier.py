"""
Real-time dual zone curve applicator for Marvel Rivals.
Reads physical controller input and applies dual zone curve transformation.
"""

import pygame
import json
import os
import time
import numpy as np
from pathlib import Path

def dual_zone_curve(input_val, deadzone, midzone, sensitivity):
    """Apply dual zone response curve to input value."""
    val = abs(input_val)
    
    if val < deadzone:
        return 0
    elif val < midzone:
        normalized = (val - deadzone) / (midzone - deadzone)
        output = normalized * 0.5
    else:
        normalized = (val - midzone) / (1.0 - midzone)
        output = 0.5 + normalized * 0.5 * sensitivity
    
    return output * np.sign(input_val)

def get_latest_curve_params():
    """Extract curve parameters from analyzer output."""
    # Default params
    params = {
        'axis_0': {'deadzone': 0.05, 'midzone': 0.5, 'sensitivity': 1.2},
        'axis_1': {'deadzone': 0.05, 'midzone': 0.5, 'sensitivity': 1.2},
        'axis_2': {'deadzone': 0.05, 'midzone': 0.5, 'sensitivity': 1.0},
        'axis_3': {'deadzone': 0.05, 'midzone': 0.5, 'sensitivity': 1.0},
    }
    
    # Try to extract from latest session
    if os.path.exists('data'):
        files = [f for f in os.listdir('data') if f.startswith('session_') and f.endswith('.csv')]
        if files:
            latest = max(files)
            # Parse the session file to calculate params
            import pandas as pd
            df = pd.read_csv(f'data/{latest}')
            
            for axis_idx in range(4):
                axis_col = f'axis_{axis_idx}'
                if axis_col in df.columns:
                    axis_data = df[axis_col].values
                    
                    # Deadzone from noise
                    center_data = axis_data[np.abs(axis_data) < 0.15]
                    deadzone = np.std(center_data) * 2.5 if len(center_data) > 0 else 0.05
                    deadzone = np.clip(deadzone, 0.02, 0.20)
                    
                    # Midzone from distribution
                    midzone = np.percentile(np.abs(axis_data[np.abs(axis_data) > deadzone]), 60)
                    midzone = np.clip(midzone, deadzone + 0.1, 0.7)
                    
                    # Sensitivity
                    outer_data = axis_data[np.abs(axis_data) > midzone]
                    sensitivity = 1.2 if len(outer_data) > 0 else 1.0
                    
                    params[axis_col] = {
                        'deadzone': float(deadzone),
                        'midzone': float(midzone),
                        'sensitivity': float(sensitivity)
                    }
    
    return params

def monitor_controller():
    """Monitor controller input and apply dual zone curve."""
    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() == 0:
        print("Error: No controller found!")
        return
    
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    print(f"Controller: {joystick.get_name()}")
    print(f"Axes: {joystick.get_numaxes()}")
    
    # Get curve parameters
    params = get_latest_curve_params()
    
    print("\nDual Zone Curve Parameters:")
    for axis, param in params.items():
        print(f"  {axis}: DZ={param['deadzone']:.4f}, MZ={param['midzone']:.4f}, Sens={param['sensitivity']:.2f}")
    
    print("\nMonitoring controller input (press Ctrl+C to exit)...")
    print("Raw Input -> Processed Output:\n")
    
    last_print = time.time()
    
    try:
        while True:
            pygame.event.pump()
            
            # Get axis values
            num_axes = min(joystick.get_numaxes(), 4)
            raw_values = [joystick.get_axis(i) for i in range(num_axes)]
            
            # Apply dual zone curve
            processed_values = []
            for i, raw_val in enumerate(raw_values):
                axis_col = f'axis_{i}'
                param = params.get(axis_col, params['axis_0'])
                processed = dual_zone_curve(raw_val, 
                                          param['deadzone'], 
                                          param['midzone'], 
                                          param['sensitivity'])
                processed_values.append(processed)
            
            # Print periodically
            if time.time() - last_print > 0.1:
                print(f"\rAxis 0: {raw_values[0]:6.3f} -> {processed_values[0]:6.3f} | " + 
                      f"Axis 1: {raw_values[1]:6.3f} -> {processed_values[1]:6.3f}", end='')
                last_print = time.time()
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
    finally:
        pygame.quit()

def generate_config_file():
    """Generate a config file that can be used by other applications."""
    params = get_latest_curve_params()
    
    config = {
        'version': '1.0',
        'type': 'dual_zone_curve',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'curves': params
    }
    
    config_path = 'data/curve_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config saved to {config_path}")
    return config_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'config':
        generate_config_file()
    else:
        monitor_controller()
