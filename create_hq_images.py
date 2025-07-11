import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def create_eye_aspect_ratio_diagram():
    """Create a high-quality EAR diagram"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left subplot - Eye landmarks
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.set_aspect('equal')
    
    # Draw eye shape
    eye_x = [1, 2, 3, 4, 5, 6]
    eye_y_top = [3, 4, 4.2, 4.2, 4, 3]
    eye_y_bottom = [3, 2.8, 2.6, 2.6, 2.8, 3]
    
    ax1.plot(eye_x, eye_y_top, 'b-', linewidth=3, label='Upper eyelid')
    ax1.plot(eye_x, eye_y_bottom, 'r-', linewidth=3, label='Lower eyelid')
    
    # Mark landmark points with accurate dlib eye indices
    eye_landmarks = [(1, 3), (2, 4), (3, 4.2), (4, 4.2), (5, 4), (6, 3)]  # Outline
    
    for i, (x, y) in enumerate(eye_landmarks):
        ax1.plot(x, y, 'ko', markersize=10)
        ax1.annotate(f'P{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='darkblue')
    
    # Draw measurement lines with proper EAR calculation
    # Vertical distances
    ax1.plot([2, 5], [4, 2.8], 'g--', linewidth=3, alpha=0.8)
    ax1.plot([3, 4], [4.2, 2.6], 'g--', linewidth=3, alpha=0.8)
    # Horizontal distance
    ax1.plot([1, 6], [3, 3], 'orange', linewidth=4, alpha=0.8)
    
    # Add annotations with arrows
    ax1.annotate('Vertical Distance A\n|P2 - P6|', xy=(2.5, 3.4), xytext=(7.5, 5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    ax1.annotate('Vertical Distance B\n|P3 - P5|', xy=(3.5, 3.4), xytext=(7.5, 4),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    ax1.annotate('Horizontal Distance C\n|P1 - P4|', xy=(3.5, 3), xytext=(7.5, 2),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                fontsize=11, color='orange', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="moccasin", alpha=0.7))
    
    ax1.set_title('Eye Aspect Ratio (EAR) Measurement', fontsize=16, fontweight='bold')
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=12)
    
    # Right subplot - EAR formula and values
    ax2.text(0.5, 0.85, 'EAR Formula:', transform=ax2.transAxes, 
             fontsize=18, fontweight='bold', ha='center')
    
    ax2.text(0.5, 0.7, r'$EAR = \frac{A + B}{2 \times C}$',
             transform=ax2.transAxes, fontsize=20, ha='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    ax2.text(0.5, 0.6, r'$EAR = \frac{|P_2 - P_6| + |P_3 - P_5|}{2 \times |P_1 - P_4|}$',
             transform=ax2.transAxes, fontsize=14, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8))
    
    # EAR values for different states
    ax2.text(0.5, 0.45, 'EAR Threshold Values:', transform=ax2.transAxes,
             fontsize=16, fontweight='bold', ha='center')
    
    # Create colored boxes for different states
    states = [
        ('Eyes Open', '> 0.25', 'green', 'Alert state'),
        ('Blinking', '0.15 - 0.25', 'orange', 'Normal blink'),
        ('Drowsy/Closed', '< 0.15', 'red', 'Drowsy state')
    ]
    
    y_pos = 0.35
    for state, ear_range, color, desc in states:
        # Create colored rectangle
        rect = Rectangle((0.1, y_pos-0.02), 0.8, 0.06, 
                        facecolor=color, alpha=0.3, transform=ax2.transAxes)
        ax2.add_patch(rect)
        
        ax2.text(0.15, y_pos, f'{state}:', transform=ax2.transAxes,
                fontsize=13, fontweight='bold', ha='left')
        ax2.text(0.4, y_pos, f'EAR {ear_range}', transform=ax2.transAxes,
                fontsize=13, ha='left', color=color, fontweight='bold')
        ax2.text(0.7, y_pos, f'({desc})', transform=ax2.transAxes,
                fontsize=11, ha='left', style='italic', color='gray')
        y_pos -= 0.1
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('eye_aspect_ratio_hq.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def create_face_detection_diagram():
    """Create a high-quality face detection diagram with accurate 68-point landmarks"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Accurate 68-point facial landmarks coordinates (normalized)
    # These are the actual landmark positions as used in dlib
    landmarks_68 = [
        # Jaw line (0-16)
        (2.2, 3.8), (2.3, 4.5), (2.5, 5.2), (2.8, 5.9), (3.2, 6.5),
        (3.8, 7.0), (4.5, 7.3), (5.2, 7.5), (5.8, 7.5), (6.5, 7.3),
        (7.2, 7.0), (7.8, 6.5), (8.2, 5.9), (8.5, 5.2), (8.7, 4.5),
        (8.8, 3.8), (9.0, 3.0),
        
        # Right eyebrow (17-21)
        (3.0, 4.8), (3.5, 4.5), (4.2, 4.4), (4.8, 4.6), (5.3, 4.9),
        
        # Left eyebrow (22-26)
        (5.7, 4.9), (6.2, 4.6), (6.8, 4.4), (7.5, 4.5), (8.0, 4.8),
        
        # Nose bridge (27-30)
        (5.5, 5.2), (5.5, 5.6), (5.5, 6.0), (5.5, 6.4),
        
        # Lower nose (31-35)
        (5.0, 6.6), (5.2, 6.8), (5.5, 6.9), (5.8, 6.8), (6.0, 6.6),
        
        # Right eye (36-41)
        (3.5, 5.8), (3.8, 5.6), (4.2, 5.6), (4.6, 5.8), (4.2, 6.0), (3.8, 6.0),
        
        # Left eye (42-47)
        (6.4, 5.8), (6.8, 5.6), (7.2, 5.6), (7.5, 5.8), (7.2, 6.0), (6.8, 6.0),
        
        # Outer lip (48-59)
        (4.8, 7.8), (5.0, 7.6), (5.2, 7.5), (5.4, 7.6), (5.6, 7.6),
        (5.8, 7.6), (6.0, 7.8), (5.8, 8.0), (5.6, 8.2), (5.4, 8.2),
        (5.2, 8.2), (5.0, 8.0),
        
        # Inner lip (60-67)
        (4.9, 7.8), (5.2, 7.7), (5.4, 7.7), (5.6, 7.7), (5.8, 7.8),
        (5.6, 7.9), (5.4, 7.9), (5.2, 7.9)
    ]
    
    # Color mapping for different facial regions
    colors = {
        'jaw': '#1f77b4',      # Blue
        'right_eyebrow': '#ff7f0e',  # Orange
        'left_eyebrow': '#ff7f0e',   # Orange
        'nose_bridge': '#2ca02c',    # Green
        'nose_lower': '#2ca02c',     # Green
        'right_eye': '#d62728',      # Red
        'left_eye': '#d62728',       # Red
        'outer_lip': '#9467bd',      # Purple
        'inner_lip': '#8c564b'       # Brown
    }
    
    # Define regions
    regions = {
        'jaw': (0, 17),
        'right_eyebrow': (17, 22),
        'left_eyebrow': (22, 27),
        'nose_bridge': (27, 31),
        'nose_lower': (31, 36),
        'right_eye': (36, 42),
        'left_eye': (42, 48),
        'outer_lip': (48, 60),
        'inner_lip': (60, 68)
    }
    
    # Draw landmarks by region
    for region, (start, end) in regions.items():
        color = colors[region]
        points = landmarks_68[start:end]
        
        # Draw points
        for i, (x, y) in enumerate(points):
            ax.plot(x, y, 'o', color=color, markersize=6, markeredgecolor='black', 
                   markeredgewidth=0.5)
            # Add point numbers for key landmarks
            if region in ['right_eye', 'left_eye'] or i % 3 == 0:
                ax.annotate(f'{start + i}', (x, y), xytext=(3, 3), 
                           textcoords='offset points', fontsize=8, 
                           color='darkblue', fontweight='bold')
        
        # Connect points to show structure
        if region == 'jaw':
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
        
        elif region in ['right_eyebrow', 'left_eyebrow']:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
        
        elif region == 'nose_bridge':
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
        
        elif region == 'nose_lower':
            x_coords = [p[0] for p in points] + [points[0][0]]
            y_coords = [p[1] for p in points] + [points[0][1]]
            ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
        
        elif region in ['right_eye', 'left_eye']:
            x_coords = [p[0] for p in points] + [points[0][0]]
            y_coords = [p[1] for p in points] + [points[0][1]]
            ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
        
        elif region == 'outer_lip':
            x_coords = [p[0] for p in points] + [points[0][0]]
            y_coords = [p[1] for p in points] + [points[0][1]]
            ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
        
        elif region == 'inner_lip':
            x_coords = [p[0] for p in points] + [points[0][0]]
            y_coords = [p[1] for p in points] + [points[0][1]]
            ax.plot(x_coords, y_coords, color=color, linewidth=1, alpha=0.7)
    
    # Add face detection bounding box
    face_box = Rectangle((2.0, 2.5), 7.2, 6.5, fill=False, color='lime', 
                        linewidth=3, linestyle='--', alpha=0.8)
    ax.add_patch(face_box)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['jaw'], 
                  markersize=8, label='Jaw (0-16)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['right_eyebrow'], 
                  markersize=8, label='Eyebrows (17-26)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['nose_bridge'], 
                  markersize=8, label='Nose (27-35)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['right_eye'], 
                  markersize=8, label='Eyes (36-47)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['outer_lip'], 
                  markersize=8, label='Lips (48-67)'),
        plt.Line2D([0], [0], color='lime', linewidth=3, linestyle='--', 
                  label='Face Detection Box')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Highlight eye regions for drowsiness detection
    # Right eye highlight
    right_eye_points = landmarks_68[36:42]
    right_eye_x = [p[0] for p in right_eye_points]
    right_eye_y = [p[1] for p in right_eye_points]
    ax.fill(right_eye_x, right_eye_y, color='red', alpha=0.2)
    
    # Left eye highlight
    left_eye_points = landmarks_68[42:48]
    left_eye_x = [p[0] for p in left_eye_points]
    left_eye_y = [p[1] for p in left_eye_points]
    ax.fill(left_eye_x, left_eye_y, color='red', alpha=0.2)
    
    # Add annotations
    ax.text(5.5, 1.5, 'dlib 68-Point Facial Landmark Detection', 
            fontsize=18, fontweight='bold', ha='center')
    
    ax.text(9.5, 8.5, 'Face Detection\nBounding Box', fontsize=11, color='lime', 
            fontweight='bold', ha='center')
    
    ax.text(4.0, 5.3, 'Right Eye\n(36-41)', fontsize=10, color='red', 
            fontweight='bold', ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.text(7.0, 5.3, 'Left Eye\n(42-47)', fontsize=10, color='red', 
            fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.text(5.5, 0.8, 'Eye regions highlighted for drowsiness detection', 
            fontsize=12, ha='center', style='italic', color='red')
    
    ax.set_xlim(1, 10)
    ax.set_ylim(0, 9.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Accurate 68-Point Facial Landmark Detection', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X Coordinate (normalized)', fontsize=12)
    ax.set_ylabel('Y Coordinate (normalized)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('face_landmarks_hq.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def create_eye_states_diagram():
    """Create a high-quality eye states comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Define eye states
    states = [
        ('Open Eyes', 0.35, 'green'),
        ('Blinking', 0.15, 'orange'), 
        ('Drowsy/Closed', 0.08, 'red')
    ]
    
    for i, (state, ear_val, color) in enumerate(states):
        ax = axes[i]
        
        # Create eye shape based on EAR value
        if ear_val > 0.3:  # Open
            eye_y_top = [3, 4, 4.2, 4.2, 4, 3]
            eye_y_bottom = [3, 2.8, 2.6, 2.6, 2.8, 3]
        elif ear_val > 0.12:  # Blinking
            eye_y_top = [3, 3.5, 3.7, 3.7, 3.5, 3]
            eye_y_bottom = [3, 2.9, 2.8, 2.8, 2.9, 3]
        else:  # Closed/Drowsy
            eye_y_top = [3, 3.1, 3.2, 3.2, 3.1, 3]
            eye_y_bottom = [3, 2.95, 2.9, 2.9, 2.95, 3]
        
        eye_x = [1, 2, 3, 4, 5, 6]
        
        # Draw eye
        ax.fill_between(eye_x, eye_y_top, eye_y_bottom, 
                       alpha=0.7, color=color, label=state)
        ax.plot(eye_x, eye_y_top, color=color, linewidth=3)
        ax.plot(eye_x, eye_y_bottom, color=color, linewidth=3)
        
        # Add EAR value
        ax.text(3.5, 1.5, f'EAR = {ear_val}', fontsize=14, fontweight='bold',
                ha='center', color=color, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add state label
        ax.text(3.5, 5, state, fontsize=16, fontweight='bold', ha='center')
        
        # Add measurement lines for open eyes
        if i == 0:
            ax.plot([2, 5], [4, 2.8], 'gray', linewidth=2, alpha=0.5, linestyle='--')
            ax.plot([3, 4], [4.2, 2.6], 'gray', linewidth=2, alpha=0.5, linestyle='--')
            ax.plot([1, 6], [3, 3], 'gray', linewidth=2, alpha=0.5, linestyle='--')
        
        ax.set_xlim(0, 7)
        ax.set_ylim(1, 5.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate', fontsize=10)
        if i == 0:
            ax.set_ylabel('Y Coordinate', fontsize=10)
    
    plt.suptitle('Eye States in Drowsiness Detection', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eye_states_hq.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def create_dashboard_preview():
    """Create a dashboard preview image"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create dashboard layout
    dashboard_bg = Rectangle((0, 0), 12, 9, facecolor='#2c2c2c', alpha=0.95)
    ax.add_patch(dashboard_bg)
    
    # Title
    ax.text(6, 8.5, 'ENHANCED DROWSINESS DETECTION DASHBOARD', 
            fontsize=20, fontweight='bold', ha='center', color='white')
    
    # Current time
    ax.text(10.5, 8.5, '14:25:30', fontsize=14, color='white', fontweight='bold')
    
    # Alertness score section
    ax.text(1, 7.8, 'Alertness Score:', fontsize=16, fontweight='bold', color='white')
    ax.text(1, 7.4, '85%', fontsize=24, fontweight='bold', color='#00ff00')
    
    # Progress bar
    progress_bg = Rectangle((1, 7), 4, 0.3, facecolor='gray', alpha=0.5)
    progress_fill = Rectangle((1, 7), 3.4, 0.3, facecolor='#00ff00', alpha=0.8)
    ax.add_patch(progress_bg)
    ax.add_patch(progress_fill)
    
    # Status text
    ax.text(1, 6.5, 'Status: ALERT', fontsize=14, fontweight='bold', color='#00ff00')
    
    # Statistics section
    ax.text(1, 6, 'Session Statistics:', fontsize=16, fontweight='bold', color='white')
    
    stats_text = [
        'Total Blinks: 45',
        'Drowsy Episodes: 0',
        'Driving Time: 25m 30s',
        'Average EAR: 0.28',
        'FPS: 28.5'
    ]
    
    for i, stat in enumerate(stats_text):
        ax.text(1, 5.6 - i*0.3, stat, fontsize=12, color='white')
    
    # EAR Graph simulation
    x_vals = np.linspace(6.5, 11, 80)
    # Create realistic EAR variation
    base_ear = 0.28
    ear_variation = 0.05 * np.sin(x_vals * 3) + 0.02 * np.random.randn(80)
    # Add some blink events
    for blink_pos in [20, 35, 55, 70]:
        if blink_pos < len(ear_variation):
            ear_variation[blink_pos-2:blink_pos+2] = -0.15
    
    y_vals = 5 + ear_variation * 5  # Scale for visualization
    
    # Graph background
    graph_bg = Rectangle((6.5, 3.5), 4.5, 3, facecolor='#1a1a1a', alpha=0.8)
    ax.add_patch(graph_bg)
    
    # Draw EAR line
    ax.plot(x_vals, y_vals, color='#00ff00', linewidth=2, label='EAR')
    
    # Threshold line
    ax.axhline(y=4.5, xmin=0.54, xmax=0.92, color='red', linestyle='--', 
               alpha=0.8, linewidth=2)
    ax.text(8.5, 4.2, 'Drowsy Threshold', fontsize=10, color='red', ha='center')
    
    # Graph title and labels
    ax.text(8.75, 6.7, 'Eye Aspect Ratio (EAR) History', 
            fontsize=14, color='white', ha='center', fontweight='bold')
    ax.text(6.7, 3.3, 'Time →', fontsize=10, color='white')
    ax.text(6.2, 5, 'EAR', fontsize=10, color='white', rotation=90, va='center')
    
    # Status indicators
    ax.text(1, 2.5, 'System Status:', fontsize=16, fontweight='bold', color='white')
    
    indicators = [
        ('CAMERA', '#00ff00', 'Connected'),
        ('SOUND', '#00ff00', 'Enabled'),
        ('RECORDING', '#ffff00', 'Active'),
        ('ALARM', '#808080', 'Standby')
    ]
    
    for i, (label, color, status) in enumerate(indicators):
        y_pos = 2.1 - i*0.25
        circle = Circle((1.3, y_pos), 0.08, facecolor=color, alpha=0.8)
        ax.add_patch(circle)
        ax.text(1.5, y_pos, f'{label}: {status}', fontsize=12, color='white', va='center')
    
    # Alert section (when not drowsy)
    alert_bg = Rectangle((6.5, 1), 4.5, 2, facecolor='#003300', alpha=0.8)
    ax.add_patch(alert_bg)
    ax.text(8.75, 2.5, 'DRIVER ALERT', fontsize=18, fontweight='bold', 
            color='#00ff00', ha='center')
    ax.text(8.75, 2.1, 'No drowsiness detected', fontsize=12, 
            color='white', ha='center')
    ax.text(8.75, 1.7, 'Drive safely!', fontsize=12, 
            color='white', ha='center', style='italic')
    
    # Add border
    border = Rectangle((0, 0), 12, 9, fill=False, color='white', linewidth=2)
    ax.add_patch(border)
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('dashboard_preview_hq.png', dpi=300, bbox_inches='tight',
                facecolor='#1a1a1a', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    print("Creating high-quality documentation images...")
    
    try:
        create_eye_aspect_ratio_diagram()
        print("✓ Created eye_aspect_ratio_hq.png")
        
        create_face_detection_diagram()
        print("✓ Created face_landmarks_hq.png")
        
        create_eye_states_diagram()
        print("✓ Created eye_states_hq.png")
        
        create_dashboard_preview()
        print("✓ Created dashboard_preview_hq.png")
        
        print("\nAll high-quality images created successfully!")
        
    except Exception as e:
        print(f"Error creating images: {e}")
        print("Please install required packages: pip install matplotlib seaborn")
