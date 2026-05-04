import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import numpy as np

def dual_zone_curve(input_val, deadzone, midzone, sensitivity):
    """
    Create a dual zone response curve for controller input.
    
    Args:
        input_val: Raw input value (-1 to 1)
        deadzone: Dead zone size (0 to 1)
        midzone: Mid zone breakpoint (0 to 1)
        sensitivity: Sensitivity multiplier for outer zone
    
    Returns:
        Processed output value (-1 to 1)
    """
    val = abs(input_val)
    
    if val < deadzone:
        # Dead zone - return 0
        return 0
    elif val < midzone:
        # Linear zone - scale from deadzone to midzone
        normalized = (val - deadzone) / (midzone - deadzone)
        output = normalized * 0.5  # Maps to 0 to 0.5
    else:
        # Outer zone - accelerated response with sensitivity
        normalized = (val - midzone) / (1.0 - midzone)
        output = 0.5 + normalized * 0.5 * sensitivity  # Maps from 0.5 to 1.0
    
    # Return with original sign
    return output * np.sign(input_val)

def analyze_and_create_curve(df, axis_col):
    """Analyze recorded data and create optimal dual zone curve."""
    
    # Calculate statistics
    axis_data = df[axis_col].values
    
    # Determine dead zone from noise around center
    center_data = axis_data[np.abs(axis_data) < 0.15]
    deadzone = np.std(center_data) * 2.5 if len(center_data) > 0 else 0.05
    deadzone = np.clip(deadzone, 0.02, 0.20)
    
    # Determine mid zone from data distribution
    outer_values = np.abs(axis_data[np.abs(axis_data) > deadzone])
    if len(outer_values) > 0:
        midzone = np.percentile(outer_values, 60)
        midzone = np.clip(midzone, deadzone + 0.1, 0.7)
    else:
        midzone = np.clip(deadzone + 0.1, 0.1, 0.7)
    
    # Calculate sensitivity from outer zone usage
    outer_data = axis_data[np.abs(axis_data) > midzone]
    sensitivity = 1.2 if len(outer_data) > 0 else 1.0
    
    return deadzone, midzone, sensitivity

# Load latest session
files = [f for f in os.listdir('data') if f.startswith('session_')]
if not files:
    print("No sessions found")
    exit()

latest = max(files)
df = pd.read_csv(f'data/{latest}')
print(f"Analyzing {latest}")
print(df.head())

# Plot axes over time
for col in df.columns[1:]:
    if 'axis' in col:
        plt.figure(figsize=(10, 6))
        plt.plot(df['timestamp'] - df['timestamp'].min(), df[col])
        plt.title(f'{col} over time')
        plt.xlabel('Time (s)')
        plt.ylabel('Axis value')
        plt.savefig(f'data/{col}_plot.png')
        plt.close()

# Create dual zone curves for main axes (axis_0 and axis_1 are typically left stick X and Y)
for axis_idx in [0, 1]:
    axis_col = f'axis_{axis_idx}'
    if axis_col not in df.columns:
        continue
    
    # Analyze and get curve parameters
    deadzone, midzone, sensitivity = analyze_and_create_curve(df, axis_col)
    
    # Create response curve visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Input distribution
    ax1.hist(df[axis_col], bins=50, alpha=0.7, color='blue')
    ax1.axvline(deadzone, color='red', linestyle='--', label=f'Dead Zone: ±{deadzone:.3f}')
    ax1.axvline(-deadzone, color='red', linestyle='--')
    ax1.axvline(midzone, color='orange', linestyle='--', label=f'Mid Zone: ±{midzone:.3f}')
    ax1.axvline(-midzone, color='orange', linestyle='--')
    ax1.set_title(f'{axis_col} - Input Value Distribution')
    ax1.set_xlabel('Input Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Dual zone response curve
    input_range = np.linspace(-1, 1, 200)
    output_range = [dual_zone_curve(val, deadzone, midzone, sensitivity) for val in input_range]
    
    ax2.plot(input_range, output_range, 'b-', linewidth=2, label='Dual Zone Curve')
    ax2.plot(input_range, input_range, 'k--', alpha=0.3, label='Linear (no processing)')
    ax2.fill_between(input_range, -deadzone, deadzone, alpha=0.2, color='red', label='Dead Zone')
    ax2.fill_between([-midzone, midzone], -1, 1, alpha=0.1, color='orange', label='Mid Zone')
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_title(f'{axis_col} - Dual Zone Response Curve')
    ax2.set_xlabel('Raw Input')
    ax2.set_ylabel('Processed Output')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'data/{axis_col}_dual_zone_curve.png', dpi=150)
    plt.close()
    
    print(f"\n{axis_col} Curve Parameters:")
    print(f"  Dead Zone: ±{deadzone:.4f}")
    print(f"  Mid Zone: ±{midzone:.4f}")
    print(f"  Sensitivity: {sensitivity:.2f}")
    
    # Apply curve to data and show before/after
    processed = df[axis_col].apply(lambda x: dual_zone_curve(x, deadzone, midzone, sensitivity))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    time_data = df['timestamp'] - df['timestamp'].min()
    
    ax1.plot(time_data, df[axis_col], 'b-', alpha=0.7, label='Raw Input')
    ax1.fill_between([-deadzone, deadzone], -1, 1, alpha=0.1, color='red', label='Dead Zone')
    ax1.set_ylabel('Raw Input Value')
    ax1.set_title(f'{axis_col} - Before & After Processing')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(time_data, processed, 'g-', alpha=0.7, label='Processed Output')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Processed Output Value')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'data/{axis_col}_before_after.png', dpi=150)
    plt.close()

print("\nAnalysis complete. Plots saved in data/")
print("Generated files:")
print("  - *_plot.png: Input over time")
print("  - *_dual_zone_curve.png: Response curve visualization")
print("  - *_before_after.png: Before/after processing comparison")

# Generate Marvel Rivals settings
print("\n" + "="*60)
print("MARVEL RIVALS - RECOMMENDED CONTROLLER SETTINGS")
print("="*60)

if 'axis_0' in df.columns and 'axis_1' in df.columns:
    # Average the parameters from both stick axes
    params_0 = {'deadzone': 0.05, 'midzone': 0.5, 'sensitivity': 1.2}
    params_1 = {'deadzone': 0.05, 'midzone': 0.5, 'sensitivity': 1.2}
    
    for axis_idx in [0, 1]:
        axis_col = f'axis_{axis_idx}'
        deadzone, midzone, sensitivity = analyze_and_create_curve(df, axis_col)
        
        if axis_idx == 0:
            params_0 = {'deadzone': deadzone, 'midzone': midzone, 'sensitivity': sensitivity}
        else:
            params_1 = {'deadzone': deadzone, 'midzone': midzone, 'sensitivity': sensitivity}
    
    # Average parameters
    avg_deadzone = (params_0['deadzone'] + params_1['deadzone']) / 2
    avg_sensitivity = (params_0['sensitivity'] + params_1['sensitivity']) / 2
    
    # Convert to Marvel Rivals Dual Zone S Curve scale (0-100 sliders)
    custom_minimum_range = int(np.clip(100 - round(avg_deadzone * 100), 1, 99))
    custom_maximum_range = max(custom_minimum_range + 1,
                               int(np.clip(100 - round(params_0['midzone'] * 100), 1, 100)))
    # Minimum curve statics should stay below the maximum dual-zone curve
    minimum_curve_statics = int(np.clip(50 - round((avg_sensitivity - 1.0) * 20), 1, 98))
    custom_maximum_dual_zone_curve = max(minimum_curve_statics + 1,
                                         int(np.clip(50 + round((avg_sensitivity - 1.0) * 50), 1, 100)))
    aim_ease_deadzone = int(np.clip(round(avg_deadzone * 100), 0, 100))
    eye_gaze_deadzone = int(np.clip(round(avg_deadzone * 100), 0, 100))
    horizontal_deadzone_boost = int(np.clip(round((avg_sensitivity - 1.0) * 100), 0, 100))
    
    print("\nSTICK SETTINGS:")
    print(f"  Horizontal Sensitivity: 200")
    print(f"  Vertical Sensitivity: 200")
    print(f"\nADVANCED SETTINGS (Dual Zone S Curve):")
    print(f"  Aim Sensitivity Curve Type: Dual Zone S Curve")
    print(f"  Custom Minimum Range: {custom_minimum_range}")
    print(f"  Custom Maximum Range: {custom_maximum_range}")
    print(f"  Minimum Curve Statics: {minimum_curve_statics}")
    print(f"  Custom Maximum Dual-zone Curve: {custom_maximum_dual_zone_curve}")
    print(f"  Eye-Gaze Targeting Minimum Input Deadzone: {aim_ease_deadzone}")
    print(f"  Eye-Gaze Targeting Maximum Input Deadzone: {eye_gaze_deadzone}")
    print(f"  Horizontal Max Deadzone Sensitivity Boost: {horizontal_deadzone_boost}")
    
    print(f"\nCALCULATED FROM YOUR SESSION:")
    print(f"  Average Deadzone: ±{avg_deadzone:.4f}")
    print(f"  Average Sensitivity Multiplier: {avg_sensitivity:.2f}x")
    print(f"  Left Stick - DZ: ±{params_0['deadzone']:.4f}, Sens: {params_0['sensitivity']:.2f}x")
    print(f"  Right Stick - DZ: ±{params_1['deadzone']:.4f}, Sens: {params_1['sensitivity']:.2f}x")
    
    print("\nHOW TO APPLY IN MARVEL RIVALS:")
    print("  1. Open Settings > Controls > Gamepad")
    print("  2. Expand 'Advanced' section")
    print("  3. Set 'Aim Sensitivity Curve Type' to 'Dual Zone S Curve'")
    print(f"  4. Set 'Custom Minimum Range' to {custom_minimum_range}")
    print(f"  5. Set 'Custom Maximum Range' to {custom_maximum_range}")
    print(f"  6. Set 'Minimum Curve Statics' to {minimum_curve_statics}")
    print(f"  7. Set 'Custom Maximum Dual-zone Curve' to {custom_maximum_dual_zone_curve}")
    print(f"  8. Set 'Eye-Gaze Targeting Minimum Input Deadzone' to {aim_ease_deadzone}")
    print(f"  9. Set 'Eye-Gaze Targeting Maximum Input Deadzone' to {eye_gaze_deadzone}")
    print(f" 10. Set 'Horizontal Max Deadzone Sensitivity Boost' to {horizontal_deadzone_boost}")
    print(" 11. Apply and test!")
    
    # Save to config file
    rivals_config = {
        'horizontal_sensitivity': 200,
        'vertical_sensitivity': 200,
        'aim_curve_type': 'Dual Zone S Curve',
        'custom_minimum_range': custom_minimum_range,
        'custom_maximum_range': custom_maximum_range,
        'minimum_curve_statics': minimum_curve_statics,
        'custom_maximum_dual_zone_curve': custom_maximum_dual_zone_curve,
        'eye_gaze_targeting_minimum_input_deadzone': aim_ease_deadzone,
        'eye_gaze_targeting_maximum_input_deadzone': eye_gaze_deadzone,
        'horizontal_max_deadzone_sensitivity_boost': horizontal_deadzone_boost,
        'calibration_data': {
            'average_deadzone': float(avg_deadzone),
            'average_sensitivity': float(avg_sensitivity),
            'left_stick_deadzone': float(params_0['deadzone']),
            'left_stick_sensitivity': float(params_0['sensitivity']),
            'right_stick_deadzone': float(params_1['deadzone']),
            'right_stick_sensitivity': float(params_1['sensitivity']),
        }
    }
    
    import json
    rivals_config_path = 'data/marvel_rivals_settings.json'
    with open(rivals_config_path, 'w') as f:
        json.dump(rivals_config, f, indent=2)
    
    print(f"\n✓ Settings saved to: {rivals_config_path}")

print("\n" + "="*60)