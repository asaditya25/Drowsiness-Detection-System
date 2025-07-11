# 🚗 Enhanced Driver Drowsiness Detection System

A sophisticated real-time driver drowsiness detection system with an advanced dashboard interface, featuring intelligent alertness monitoring, smart alarm management, and comprehensive analytics.

## ✨ Key Features

### 🎯 **Advanced Detection**
- Real-time face detection using dlib's 68-point facial landmark model
- Adaptive eye aspect ratio (EAR) thresholding for improved accuracy
- Intelligent blink pattern analysis with false positive reduction
- Multi-factor alertness scoring system

### 📊 **Enhanced Dashboard**
- **Live Alertness Score**: Real-time driver alertness percentage (0-100%)
- **EAR Graph**: Visual representation of eye aspect ratio over time
- **Statistics Panel**: Comprehensive driving session analytics
- **Status Indicators**: Real-time system status monitoring
- **Configurable Layout**: Choose dashboard position (right, left, top, bottom)

### 🔔 **Smart Alarm System**
- **Auto-Stop Feature**: Alarm automatically stops after 3 seconds of open eyes
- **Visual Countdown**: Shows remaining time before alarm stops
- **Intelligent Reset**: Timer resets if eyes close during countdown
- **Cross-Platform Audio**: Windows (winsound) and Linux (aplay) support

### 📈 **Analytics & Monitoring**
- Session statistics with driving time tracking
- Drowsy episode counting and analysis
- Blink rate monitoring and assessment
- Performance metrics and FPS monitoring
- Exportable statistics in JSON format

## 🛠️ Installation

### Prerequisites
1. **Python 3.7+** with pip installed
2. **Webcam** (USB or built-in camera)
3. **dlib facial landmark model**:
   ```bash
   # Download the model file (95MB)
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   # Extract it
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   ```

### Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# For conda users (recommended for dlib):
conda install dlib opencv scipy numpy playsound -c conda-forge
```

### Dependencies
- **opencv-python**: Computer vision and image processing
- **dlib**: Face detection and landmark extraction
- **scipy**: Scientific computing for distance calculations
- **numpy**: Numerical operations and array handling
- **playsound**: Cross-platform audio alerts

## 🚀 Usage

### Basic Usage
```bash
# Run with default settings
python enhanced_dashboard_detector.py

# Run with custom configuration
python enhanced_dashboard_detector.py --config config.json

# Run without sound alerts
python enhanced_dashboard_detector.py --no-sound

# Run without video recording
python enhanced_dashboard_detector.py --no-video

# Set custom EAR threshold
python enhanced_dashboard_detector.py --threshold 0.23

# Position dashboard on the left
python enhanced_dashboard_detector.py --dashboard-position left
```

### Available Command Line Options
- `--config`: Path to configuration file
- `--no-sound`: Disable sound alerts
- `--no-video`: Disable video recording
- `--no-dashboard`: Disable dashboard display
- `--threshold`: Set custom EAR threshold (0.1-0.4)
- `--dashboard-position`: Set dashboard position (right/left/top/bottom)

### Keyboard Controls
- **ESC**: Exit application
- **R**: Reset detection state and clear alarms
- **S**: Save current statistics to file
- **C**: Recalibrate the detection system
- **D**: Toggle dashboard on/off

## ⚙️ Configuration

The system uses a JSON configuration file (`config.json`) for customization:

```json
{
  "ear_threshold": 0.25,           // Eye closure threshold
  "blink_time": 0.15,              // Normal blink duration (seconds)
  "drowsy_time": 1.5,              // Drowsy detection time (seconds)
  "alarm_stop_delay": 3.0,         // Auto-stop alarm delay (seconds)
  "dashboard_enabled": true,        // Enable/disable dashboard
  "dashboard_position": "right",    // Dashboard position
  "show_ear_graph": true,          // Show EAR graph
  "confidence_threshold": 0.8,     // Face detection confidence
  "adaptive_threshold": true,      // Use adaptive EAR thresholding
  "sound_enabled": true,           // Enable sound alerts
  "video_recording": true,         // Enable video recording
  "alertness_threshold": 70        // Alertness warning threshold
}
```

## 📊 Dashboard Components

### 1. **Alertness Score**
- **Green (80-100%)**: Alert and safe
- **Yellow (60-79%)**: Slightly tired, monitor closely
- **Red (0-59%)**: Drowsy, take action immediately

### 2. **Status Indicators**
- **CAMERA**: Camera connection status
- **SOUND**: Audio alerts status
- **RECORDING**: Video recording status
- **ALARM**: Current alarm state

### 3. **Statistics Panel**
- **Blinks**: Total blinks in current session
- **Drowsy Episodes**: Number of drowsiness events
- **Driving Time**: Current session duration
- **FPS**: System performance indicator

### 4. **EAR Graph**
- Real-time eye aspect ratio visualization
- Red threshold line for reference
- Historical data for pattern analysis

## 🔬 Algorithm Details

### Eye Aspect Ratio (EAR) Calculation
The system calculates the eye aspect ratio using the formula:
```
EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
```
Where p1-p6 are the 6 eye landmark points.

### Alertness Scoring
The alertness score is calculated based on:
- Recent drowsy episodes (weight: 30%)
- Average EAR values (weight: 25%)
- Blink rate analysis (weight: 20%)
- Current drowsy state (weight: 25%)

### Adaptive Thresholding
The system automatically adjusts the EAR threshold based on:
- Personal baseline EAR values
- Lighting conditions
- Camera quality and positioning

## 📁 Project Structure

```
Drowsiness Detection System/
├── enhanced_dashboard_detector.py    # Main enhanced detector
├── enhanced_drowsiness_detector.py   # Alternative detector
├── blinkDetect.py                   # Original detector
├── config.json                      # Configuration file
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── .gitignore                       # Git ignore rules
├── alarm.wav                        # Alert sound file
├── shape_predictor_68_face_landmarks.dat  # dlib model (95MB)
├── eye.PNG                          # Documentation image
├── eye_aspect_ratio.PNG             # Documentation image
└── face.PNG                         # Documentation image
```

## 🎯 Performance Optimization

### Hardware Recommendations
- **CPU**: Intel i5 or AMD Ryzen 5 (minimum)
- **RAM**: 8GB+ (for optimal performance)
- **Camera**: 720p+ resolution for better face detection
- **Storage**: SSD recommended for faster model loading

### Software Optimization
- Use conda environment for better dlib performance
- Adjust `face_downsample_ratio` for performance vs accuracy
- Lower `resize_height` for faster processing
- Disable video recording if not needed

## 🔧 Troubleshooting

### Common Issues

1. **"Could not find dlib face landmark model"**
   - Download the model file from the provided link
   - Place it in the project directory
   - Ensure the file is not corrupted

2. **"Could not open camera"**
   - Check camera permissions
   - Ensure camera is not used by other applications
   - Try different camera indices (0, 1, 2, etc.)

3. **Poor detection accuracy**
   - Improve lighting conditions
   - Position camera at eye level
   - Adjust EAR threshold in configuration
   - Enable adaptive thresholding

4. **Low FPS performance**
   - Reduce `resize_height` value
   - Increase `face_downsample_ratio`
   - Close other applications
   - Use a faster computer

### Performance Tuning
```json
// For better performance (lower accuracy)
{
  "face_downsample_ratio": 2.0,
  "resize_height": 360,
  "calibration_frames": 50
}

// For better accuracy (lower performance)
{
  "face_downsample_ratio": 1.2,
  "resize_height": 720,
  "calibration_frames": 150
}
```

## 📊 Statistics Export

The system automatically saves statistics in JSON format:
```json
{
  "session_start": "2025-01-15T10:30:00",
  "total_blinks": 45,
  "drowsy_episodes": 2,
  "driving_time": 1800,
  "alertness_score": 85,
  "avg_blink_rate": 15.5
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **dlib**: For the excellent face detection library
- **OpenCV**: For computer vision capabilities
- **SciPy**: For mathematical computations
- **Research Papers**: Various studies on drowsiness detection algorithms

## 📞 Support

For support, issues, or feature requests:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration options

---

**⚠️ Safety Notice**: This system is designed to assist drivers but should not be the sole safety measure. Always prioritize getting adequate rest before driving and pull over safely if you feel drowsy.

**🚗 Drive Safe, Stay Alert!**
