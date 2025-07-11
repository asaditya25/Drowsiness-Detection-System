import dlib, sys, cv2, time, numpy as np, json, argparse, logging, warnings, math
from scipy.spatial import distance as dist
from threading import Thread, Lock
from datetime import datetime, timedelta
from collections import deque
import queue, os, platform, subprocess

warnings.filterwarnings("ignore")

class EnhancedDashboardDetector:
    def __init__(self, config_file=None):
        """Initialize the enhanced drowsiness detector with modern dashboard"""
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.detector = self.predictor = self.capture = self.vid_writer = None
        
        # State variables
        self.state = self.blink_count = self.drowsy = self.consecutive_open_eyes = 0
        self.alarm_on = False
        self.thread_status_q = queue.Queue()
        self.state_lock = Lock()
        self.alarm_start_time = self.eyes_open_time = None
        
        # Dashboard colors and dimensions
        self.dashboard_width, self.dashboard_height = 400, 300
        self.colors = {
            'bg': (40, 40, 40), 'alert': (0, 0, 255), 'safe': (0, 255, 0),
            'warning': (0, 255, 255), 'text': (255, 255, 255)
        }
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.detection_confidence = deque(maxlen=10)
        self.ear_history = deque(maxlen=100)
        self.drowsy_episodes_today = 0
        self.session_start_time = time.time()
        
        # Statistics
        self.stats = {
            'total_blinks': 0, 'drowsy_episodes': 0, 'false_alarms': 0,
            'session_start': datetime.now(), 'last_calibration': None,
            'driving_time': 0, 'max_continuous_driving': 0,
            'avg_blink_rate': 0, 'alertness_score': 100
        }
        
        self.initialize_models()
        self.initialize_camera()
        
    def load_config(self, config_file):
        """Load configuration from file or use defaults"""
        default_config = {
            'face_downsample_ratio': 1.5, 'resize_height': 460, 'ear_threshold': 0.25,
            'blink_time': 0.15, 'drowsy_time': 1.5, 'gamma': 1.5, 'calibration_frames': 100,
            'confidence_threshold': 0.8, 'adaptive_threshold': True, 'enable_logging': True,
            'log_level': 'INFO', 'sound_enabled': True, 'video_recording': True,
            'output_video_name': 'drowsiness_detection_output.avi', 'fps': 15,
            'alarm_stop_delay': 3.0, 'min_open_frames': 10, 'dashboard_enabled': True,
            'dashboard_position': 'right', 'show_ear_graph': True, 'show_statistics': True,
            'alertness_threshold': 70
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    default_config.update(json.load(f))
                print(f"Loaded configuration from {config_file}")
            except Exception as e:
                print(f"Error loading config file: {e}. Using defaults.")
        
        return default_config
        
    def setup_logging(self):
        """Setup logging configuration"""
        if self.config['enable_logging']:
            logging.basicConfig(
                level=getattr(logging, self.config['log_level'].upper()),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[logging.FileHandler('drowsiness_detection.log'), logging.StreamHandler()]
            )
    
    def initialize_models(self):
        """Initialize dlib face detector and predictor"""
        try:
            # Find model file
            model_filename = "shape_predictor_68_face_landmarks.dat"
            possible_paths = [
                os.path.join(os.path.dirname(__file__), model_filename),
                os.path.join(os.path.dirname(__file__), "models", model_filename),
                os.path.join(os.path.expanduser("~"), "Downloads", "model", model_filename),
                model_filename
            ]
            
            model_path = next((path for path in possible_paths if os.path.exists(path)), None)
            if model_path is None:
                raise FileNotFoundError("Could not find the dlib face landmark model file!")
            
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(model_path)
            
            # Eye landmark indices and gamma correction table
            self.left_eye_index = [36, 37, 38, 39, 40, 41]
            self.right_eye_index = [42, 43, 44, 45, 46, 47]
            inv_gamma = 1.0 / self.config['gamma']
            self.gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 
                                       for i in range(256)]).astype("uint8")
            
            self.logger.info(f"Successfully loaded face landmark model from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            self.cleanup_and_exit(1)
    
    def initialize_camera(self):
        """Initialize camera capture"""
        try:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                raise RuntimeError("Could not open camera")
            
            # Set camera properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            
            # Camera warm-up
            for _ in range(10):
                ret, _ = self.capture.read()
                if not ret:
                    raise RuntimeError("Camera warm-up failed")
            
            self.logger.info("Camera initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing camera: {e}")
            self.cleanup_and_exit(1)
    
    def eye_aspect_ratio(self, eye_landmarks):
        """Calculate eye aspect ratio with improved accuracy"""
        try:
            A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
            B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
            C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
            return (A + B) / (2.0 * C) if C != 0 else 0
        except Exception as e:
            self.logger.warning(f"Error calculating EAR: {e}")
            return 0
    
    def get_landmarks(self, image):
        """Extract facial landmarks with improved error handling"""
        try:
            # Resize and convert to grayscale
            img_small = cv2.resize(image, None, 
                                 fx=1.0/self.config['face_downsample_ratio'], 
                                 fy=1.0/self.config['face_downsample_ratio'], 
                                 interpolation=cv2.INTER_LINEAR)
            
            if len(img_small.shape) == 3:
                img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
            
            # Detect faces and get largest one
            faces = self.detector(img_small, 1)
            if len(faces) == 0:
                return None, 0
            
            largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Scale rectangle back to original size
            ratio = self.config['face_downsample_ratio']
            scaled_rect = dlib.rectangle(
                int(largest_face.left() * ratio), int(largest_face.top() * ratio),
                int(largest_face.right() * ratio), int(largest_face.bottom() * ratio)
            )
            
            # Get landmarks and calculate confidence
            landmarks = self.predictor(image, scaled_rect)
            points = [(p.x, p.y) for p in landmarks.parts()]
            face_area = largest_face.width() * largest_face.height()
            confidence = min(face_area / (img_small.shape[0] * img_small.shape[1] * 0.1), 1.0)
            
            return points, confidence
            
        except Exception as e:
            self.logger.warning(f"Error extracting landmarks: {e}")
            return None, 0
    
    def check_eye_status(self, landmarks):
        """Check eye status with improved reliability"""
        try:
            # Calculate EAR for both eyes
            left_eye = [landmarks[i] for i in self.left_eye_index]
            right_eye = [landmarks[i] for i in self.right_eye_index]
            avg_ear = (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)) / 2.0
            
            # Store EAR history
            self.ear_history.append(avg_ear)
            
            # Adaptive threshold
            if self.config['adaptive_threshold'] and len(self.ear_history) > 10:
                threshold = np.mean(self.ear_history) * 0.8
                threshold = max(min(threshold, 0.35), 0.2)
            else:
                threshold = self.config['ear_threshold']
            
            return (1 if avg_ear > threshold else 0), avg_ear, threshold
            
        except Exception as e:
            self.logger.warning(f"Error checking eye status: {e}")
            return 1, 0, self.config['ear_threshold']
    
    def check_blink_status(self, eye_status):
        """Check blink status with improved logic"""
        with self.state_lock:
            if self.state >= 0 and self.state <= self.false_blink_limit:
                if eye_status:
                    self.state = 0
                    self.consecutive_open_eyes += 1
                else:
                    self.state += 1
                    self.consecutive_open_eyes = 0
            elif self.state >= self.false_blink_limit and self.state < self.drowsy_limit:
                if eye_status:
                    self.blink_count += 1
                    self.stats['total_blinks'] += 1
                    self.state = 0
                    self.consecutive_open_eyes += 1
                else:
                    self.state += 1
                    self.consecutive_open_eyes = 0
            else:
                if eye_status:
                    self.state = 0
                    self.drowsy = 1
                    self.stats['drowsy_episodes'] += 1
                    self.drowsy_episodes_today += 1
                    self.blink_count += 1
                    self.consecutive_open_eyes += 1
                else:
                    self.drowsy = 1
                    self.consecutive_open_eyes = 0
            
            # Handle alarm stopping logic
            if self.alarm_on and eye_status:
                if self.eyes_open_time is None:
                    self.eyes_open_time = time.time()
                elif time.time() - self.eyes_open_time >= self.config['alarm_stop_delay']:
                    self.stop_alarm()
                    self.drowsy = 0
                    self.eyes_open_time = None
                    self.logger.info("Alarm stopped - eyes open for sufficient time")
            elif self.alarm_on and not eye_status:
                self.eyes_open_time = None
    
    def stop_alarm(self):
        """Stop the alarm system"""
        if self.alarm_on:
            self.thread_status_q.put(True)
            self.alarm_on = False
            self.eyes_open_time = None
    
    def calculate_alertness_score(self):
        """Calculate alertness score based on various factors"""
        try:
            score = 100
            
            # Deduct for drowsy episodes, low EAR, abnormal blink rate, and current drowsy state
            if self.drowsy_episodes_today > 0:
                score -= min(self.drowsy_episodes_today * 15, 50)
            
            if len(self.ear_history) > 10:
                avg_ear = np.mean(list(self.ear_history)[-30:])
                score -= 30 if avg_ear < 0.2 else 15 if avg_ear < 0.25 else 0
            
            driving_time_minutes = (time.time() - self.session_start_time) / 60
            if driving_time_minutes > 1:
                blink_rate = self.stats['total_blinks'] / driving_time_minutes
                score -= 20 if blink_rate < 10 else 10 if blink_rate > 30 else 0
            
            if self.drowsy:
                score -= 40
            
            return max(0, min(100, score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating alertness score: {e}")
            return 50
    
    def draw_dashboard(self, frame):
        """Draw enhanced dashboard with modern design"""
        try:
            if not self.config['dashboard_enabled']:
                return frame
            
            frame_h, frame_w = frame.shape[:2]
            
            # Calculate dashboard position
            if self.config['dashboard_position'] == 'right':
                extended_frame = np.zeros((frame_h, frame_w + self.dashboard_width, 3), dtype=np.uint8)
                extended_frame[:, :frame_w] = frame
                dashboard_x, dashboard_y = frame_w, 0
            elif self.config['dashboard_position'] == 'bottom':
                extended_frame = np.zeros((frame_h + self.dashboard_height, frame_w, 3), dtype=np.uint8)
                extended_frame[:frame_h, :] = frame
                dashboard_x, dashboard_y = 0, frame_h
            else:
                extended_frame = frame.copy()
                dashboard_x, dashboard_y = frame_w - self.dashboard_width, 0
            
            # Draw dashboard background and border
            cv2.rectangle(extended_frame, (dashboard_x, dashboard_y), 
                         (dashboard_x + self.dashboard_width, dashboard_y + self.dashboard_height),
                         self.colors['bg'], -1)
            cv2.rectangle(extended_frame, (dashboard_x, dashboard_y), 
                         (dashboard_x + self.dashboard_width, dashboard_y + self.dashboard_height),
                         self.colors['text'], 2)
            
            # Dashboard title and time
            cv2.putText(extended_frame, "DRIVER MONITOR", (dashboard_x + 10, dashboard_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
            cv2.putText(extended_frame, datetime.now().strftime("%H:%M:%S"), 
                       (dashboard_x + 250, dashboard_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            
            # Alertness score and bar
            alertness_score = self.calculate_alertness_score()
            score_color = (self.colors['safe'] if alertness_score > 80 else 
                          self.colors['warning'] if alertness_score > 60 else self.colors['alert'])
            
            cv2.putText(extended_frame, f"Alertness: {alertness_score:.0f}%", 
                       (dashboard_x + 10, dashboard_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2)
            
            # Alertness bar
            bar_x, bar_y = dashboard_x + 10, dashboard_y + 70
            cv2.rectangle(extended_frame, (bar_x, bar_y), (bar_x + 200, bar_y + 10), (100, 100, 100), -1)
            cv2.rectangle(extended_frame, (bar_x, bar_y), (bar_x + int(alertness_score * 2), bar_y + 10), score_color, -1)
            
            # Statistics
            stats_to_show = [
                f"Blinks: {self.stats['total_blinks']}", f"Drowsy Episodes: {self.drowsy_episodes_today}",
                f"Driving Time: {self.format_time(time.time() - self.session_start_time)}",
                f"FPS: {np.mean(self.fps_counter):.1f}" if self.fps_counter else "FPS: 0"
            ]
            
            for i, stat in enumerate(stats_to_show):
                cv2.putText(extended_frame, stat, (dashboard_x + 10, dashboard_y + 100 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            
            # EAR Graph and Status indicators
            if self.config['show_ear_graph'] and len(self.ear_history) > 1:
                self.draw_ear_graph(extended_frame, dashboard_x, dashboard_y + 200)
            
            self.draw_status_indicators(extended_frame, dashboard_x, dashboard_y + 280)
            
            return extended_frame
            
        except Exception as e:
            self.logger.warning(f"Error drawing dashboard: {e}")
            return frame
    
    def draw_ear_graph(self, frame, x, y):
        """Draw EAR (Eye Aspect Ratio) graph"""
        try:
            graph_width, graph_height = 350, 60
            
            # Graph background and border
            cv2.rectangle(frame, (x, y), (x + graph_width, y + graph_height), (60, 60, 60), -1)
            cv2.rectangle(frame, (x, y), (x + graph_width, y + graph_height), self.colors['text'], 1)
            cv2.putText(frame, "EAR History", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            
            # Draw EAR line
            if len(self.ear_history) > 1:
                ear_data = list(self.ear_history)[-60:]
                max_ear, min_ear = max(ear_data), min(ear_data)
                
                if max_ear > min_ear:
                    normalized_data = [(val - min_ear) / (max_ear - min_ear) for val in ear_data]
                    for i in range(1, len(normalized_data)):
                        x1 = x + int((i - 1) * graph_width / len(normalized_data))
                        y1 = y + graph_height - int(normalized_data[i - 1] * graph_height)
                        x2 = x + int(i * graph_width / len(normalized_data))
                        y2 = y + graph_height - int(normalized_data[i] * graph_height)
                        cv2.line(frame, (x1, y1), (x2, y2), self.colors['safe'], 2)
                    
                    # Draw threshold line
                    threshold = self.config['ear_threshold']
                    if min_ear <= threshold <= max_ear:
                        threshold_y = y + graph_height - int(((threshold - min_ear) / (max_ear - min_ear)) * graph_height)
                        cv2.line(frame, (x, threshold_y), (x + graph_width, threshold_y), self.colors['alert'], 1)
            
        except Exception as e:
            self.logger.warning(f"Error drawing EAR graph: {e}")
    
    def draw_status_indicators(self, frame, x, y):
        """Draw status indicators"""
        try:
            indicators = [
                ("CAMERA", self.colors['safe'] if self.capture and self.capture.isOpened() else self.colors['alert']),
                ("SOUND", self.colors['safe'] if self.config['sound_enabled'] else (100, 100, 100)),
                ("RECORDING", self.colors['safe'] if self.vid_writer else (100, 100, 100)),
                ("ALARM", self.colors['alert'] if self.alarm_on else (100, 100, 100))
            ]
            
            for i, (label, color) in enumerate(indicators):
                indicator_x = x + 10 + (i * 90)
                cv2.circle(frame, (indicator_x, y), 5, color, -1)
                cv2.putText(frame, label, (indicator_x + 10, y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.colors['text'], 1)
            
        except Exception as e:
            self.logger.warning(f"Error drawing status indicators: {e}")
    
    def format_time(self, seconds):
        """Format seconds into readable time"""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    
    def enhanced_preprocessing(self, image):
        """Enhanced image preprocessing for better detection"""
        try:
            # Gamma correction and histogram equalization
            gamma_corrected = cv2.LUT(image, self.gamma_table)
            
            if len(image.shape) == 3:
                yuv = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            else:
                enhanced = cv2.equalizeHist(gamma_corrected)
            
            # Noise reduction
            return cv2.bilateralFilter(enhanced, 9, 75, 75)
            
        except Exception as e:
            self.logger.warning(f"Error in preprocessing: {e}")
            return image
    
    def sound_alert(self, sound_path):
        """Improved sound alert system"""
        if not self.config['sound_enabled'] or sound_path is None:
            return
        
        try:
            abs_path = os.path.abspath(sound_path)
            
            if platform.system() == "Windows":
                import winsound
                while True:
                    if not self.thread_status_q.empty() and self.thread_status_q.get():
                        break
                    try:
                        winsound.PlaySound(abs_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                        time.sleep(1)
                    except Exception as e:
                        self.logger.error(f"Error playing sound: {e}")
                        break
            else:
                while True:
                    if not self.thread_status_q.empty() and self.thread_status_q.get():
                        break
                    try:
                        subprocess.run(["aplay", abs_path], capture_output=True, timeout=2)
                        time.sleep(0.5)
                    except Exception as e:
                        self.logger.error(f"Error playing sound: {e}")
                        break
                        
        except Exception as e:
            self.logger.error(f"Sound alert error: {e}")
    
    def calibrate(self):
        """Calibration process with progress tracking"""
        print("Starting calibration...")
        self.logger.info("Calibration started")
        
        total_time = valid_frames = 0
        calibration_frames = self.config['calibration_frames']
        
        while valid_frames < calibration_frames:
            try:
                start_time = time.time()
                ret, frame = self.capture.read()
                
                if not ret:
                    continue
                
                # Resize and preprocess frame
                height, width = frame.shape[:2]
                image_resize = np.float32(height) / self.config['resize_height']
                frame = cv2.resize(frame, None, fx=1/image_resize, fy=1/image_resize, interpolation=cv2.INTER_LINEAR)
                enhanced = self.enhanced_preprocessing(frame)
                
                # Get landmarks
                landmarks, confidence = self.get_landmarks(enhanced)
                
                if landmarks is None or confidence < self.config['confidence_threshold']:
                    cv2.putText(frame, "Please position your face properly", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Calibration: {valid_frames}/{calibration_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    valid_frames += 1
                    total_time += time.time() - start_time
                    
                    # Show progress and draw landmarks
                    progress = (valid_frames / calibration_frames) * 100
                    cv2.putText(frame, f"Calibration: {progress:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    for i in self.left_eye_index + self.right_eye_index:
                        cv2.circle(frame, landmarks[i], 2, (0, 255, 0), -1)
                
                cv2.imshow("Drowsiness Detection - Calibration", frame)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    self.cleanup_and_exit(0)
                    
            except Exception as e:
                self.logger.error(f"Error during calibration: {e}")
                continue
        
        # Calculate performance metrics
        self.spf = total_time / calibration_frames
        self.drowsy_limit = self.config['drowsy_time'] / self.spf
        self.false_blink_limit = self.config['blink_time'] / self.spf
        self.stats['last_calibration'] = datetime.now()
        
        print(f"Calibration complete! SPF: {self.spf*1000:.2f} ms, Drowsy limit: {self.drowsy_limit:.1f} frames, False blink limit: {self.false_blink_limit:.1f} frames")
        self.logger.info(f"Calibration completed - SPF: {self.spf*1000:.2f} ms")
        
        cv2.destroyWindow("Drowsiness Detection - Calibration")
    
    def run(self):
        """Main detection loop with enhanced dashboard"""
        try:
            self.calibrate()
            
            # Initialize sound path
            sound_path = None
            if self.config['sound_enabled'] and os.path.exists("alarm.wav"):
                sound_path = "alarm.wav"
            elif self.config['sound_enabled']:
                self.logger.warning("Sound file not found. Audio alerts disabled.")
            
            print("Starting drowsiness detection with enhanced dashboard...")
            self.logger.info("Detection started")
            
            while True:
                try:
                    frame_start = time.time()
                    ret, frame = self.capture.read()
                    
                    if not ret:
                        self.logger.error("Failed to read frame")
                        break
                    
                    # Resize frame
                    height, width = frame.shape[:2]
                    image_resize = np.float32(height) / self.config['resize_height']
                    frame = cv2.resize(frame, None, fx=1/image_resize, fy=1/image_resize, interpolation=cv2.INTER_LINEAR)
                    
                    # Initialize video writer if needed
                    if self.vid_writer is None and self.config['video_recording']:
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        self.vid_writer = cv2.VideoWriter(
                            self.config['output_video_name'], fourcc, 
                            self.config['fps'], (frame.shape[1], frame.shape[0]))
                    
                    # Process frame
                    enhanced = self.enhanced_preprocessing(frame)
                    landmarks, confidence = self.get_landmarks(enhanced)
                    
                    if landmarks is None or confidence < self.config['confidence_threshold']:
                        cv2.putText(frame, "Face not detected - Check lighting", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        # Eye detection and status check
                        eye_status, ear, threshold = self.check_eye_status(landmarks)
                        self.check_blink_status(eye_status)
                        
                        # Draw landmarks and face rectangle
                        for i in self.left_eye_index + self.right_eye_index:
                            cv2.circle(frame, landmarks[i], 2, (0, 255, 0), -1)
                        
                        face_rect = cv2.boundingRect(np.array(landmarks))
                        cv2.rectangle(frame, (face_rect[0], face_rect[1]), 
                                    (face_rect[0] + face_rect[2], face_rect[1] + face_rect[3]), (255, 0, 0), 2)
                        
                        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Handle drowsiness alert
                    if self.drowsy:
                        cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (50, frame.shape[0] - 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        
                        if not self.alarm_on and sound_path:
                            self.logger.warning("DROWSINESS DETECTED - ALARM TRIGGERED")
                            self.alarm_on = True
                            self.alarm_start_time = time.time()
                            # Clear queue and start alarm thread
                            while not self.thread_status_q.empty():
                                self.thread_status_q.get()
                            alarm_thread = Thread(target=self.sound_alert, args=(sound_path,))
                            alarm_thread.daemon = True
                            alarm_thread.start()
                        
                        # Show alarm countdown
                        if self.alarm_on and self.eyes_open_time is not None:
                            remaining_time = self.config['alarm_stop_delay'] - (time.time() - self.eyes_open_time)
                            if remaining_time > 0:
                                cv2.putText(frame, f"Alarm stops in: {remaining_time:.1f}s", 
                                          (50, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    # Calculate FPS and draw dashboard
                    frame_time = time.time() - frame_start
                    fps = 1.0 / frame_time if frame_time > 0 else 0
                    self.fps_counter.append(fps)
                    
                    frame_with_dashboard = self.draw_dashboard(frame)
                    cv2.imshow("Enhanced Drowsiness Detection Dashboard", frame_with_dashboard)
                    
                    # Write to video file
                    if self.vid_writer is not None:
                        self.vid_writer.write(frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                    elif key == ord('r'):  # Reset
                        self.reset_state()
                    elif key == ord('s'):  # Save stats
                        self.save_statistics()
                    elif key == ord('c'):  # Recalibrate
                        self.calibrate()
                    elif key == ord('d'):  # Toggle dashboard
                        self.config['dashboard_enabled'] = not self.config['dashboard_enabled']
                    
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    continue
            
        except KeyboardInterrupt:
            self.logger.info("Detection stopped by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
        finally:
            self.cleanup_and_exit(0)
    
    def reset_state(self):
        """Reset detection state"""
        with self.state_lock:
            self.state = self.drowsy = self.consecutive_open_eyes = 0
            self.eyes_open_time = self.alarm_start_time = None
            if self.alarm_on:
                self.thread_status_q.put(True)
                self.alarm_on = False
        self.logger.info("State reset")
        print("State reset")
    
    def save_statistics(self):
        """Save detection statistics"""
        try:
            stats_file = f"detection_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(stats_file, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                stats_copy = self.stats.copy()
                stats_copy['session_start'] = stats_copy['session_start'].isoformat()
                if stats_copy['last_calibration']:
                    stats_copy['last_calibration'] = stats_copy['last_calibration'].isoformat()
                stats_copy['alertness_score'] = self.calculate_alertness_score()
                json.dump(stats_copy, f, indent=2)
            self.logger.info(f"Statistics saved to {stats_file}")
            print(f"Statistics saved to {stats_file}")
        except Exception as e:
            self.logger.error(f"Error saving statistics: {e}")
    
    def cleanup_and_exit(self, exit_code=0):
        """Clean up resources and exit"""
        try:
            if self.alarm_on:
                self.stop_alarm()
            
            if self.capture:
                self.capture.release()
            if self.vid_writer:
                self.vid_writer.release()
            
            cv2.destroyAllWindows()
            self.save_statistics()
            self.logger.info("Application closed gracefully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
        
        sys.exit(exit_code)


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Enhanced Drowsiness Detection System with Dashboard")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--no-sound", action="store_true", help="Disable sound alerts")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable dashboard")
    parser.add_argument("--threshold", type=float, help="EAR threshold for eye closure")
    parser.add_argument("--dashboard-position", choices=['right', 'left', 'top', 'bottom'], help="Dashboard position")
    
    args = parser.parse_args()
    
    try:
        detector = EnhancedDashboardDetector(args.config)
        
        # Override config with command line arguments
        if args.no_sound:
            detector.config['sound_enabled'] = False
        if args.no_video:
            detector.config['video_recording'] = False
        if args.no_dashboard:
            detector.config['dashboard_enabled'] = False
        if args.threshold:
            detector.config['ear_threshold'] = args.threshold
        if args.dashboard_position:
            detector.config['dashboard_position'] = args.dashboard_position
        
        detector.run()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
