import cv2
import mediapipe as mp
import numpy as np
import math
import time
from datetime import datetime
from collections import deque

class VTuberAvatar:
    def __init__(self, avatar_style='cute'):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Background removal
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        
        # Avatar parameters
        self.avatar_center = (1000, 360)  # Position of avatar on screen
        self.head_rotation = [0, 0, 0]  # pitch, yaw, roll
        self.eye_open_ratio = [1.0, 1.0]  # left, right
        self.mouth_open_ratio = 0.0
        self.emotion = 'neutral'  # Current emotion
        self.avatar_style = avatar_style
        
        # Hand tracking
        self.hand_positions = {'left': None, 'right': None}
        self.hand_gestures = {'left': 'none', 'right': 'none'}
        
        # Recording
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        
        # Performance metrics
        self.fps_counter = deque(maxlen=30)
        self.current_fps = 0
        
        # Background settings
        self.background_color = (50, 150, 50)  # Green screen default
        self.use_background_removal = False
        
        # Avatar colors based on style
        self.avatar_colors = self.get_avatar_colors(avatar_style)
        
    def get_avatar_colors(self, style):
        """Get color scheme based on avatar style"""
        color_schemes = {
            'cute': {
                'skin': (255, 220, 180),
                'skin_outline': (200, 170, 140),
                'eye_white': (255, 255, 255),
                'eye_outline': (100, 100, 100),
                'iris': (100, 150, 200),
                'pupil': (50, 50, 50),
                'mouth': (255, 150, 150),
                'blush': (255, 180, 200)
            },
            'anime': {
                'skin': (255, 230, 200),
                'skin_outline': (220, 180, 150),
                'eye_white': (255, 255, 255),
                'eye_outline': (50, 50, 50),
                'iris': (50, 100, 255),
                'pupil': (20, 20, 20),
                'mouth': (255, 100, 100),
                'blush': (255, 150, 180)
            },
            'cool': {
                'skin': (200, 220, 255),
                'skin_outline': (150, 170, 200),
                'eye_white': (240, 250, 255),
                'eye_outline': (80, 100, 120),
                'iris': (100, 200, 255),
                'pupil': (30, 50, 80),
                'mouth': (180, 200, 255),
                'blush': (200, 220, 255)
            },
            'warm': {
                'skin': (255, 200, 150),
                'skin_outline': (200, 150, 100),
                'eye_white': (255, 250, 240),
                'eye_outline': (120, 100, 80),
                'iris': (150, 100, 50),
                'pupil': (80, 50, 30),
                'mouth': (255, 150, 100),
                'blush': (255, 180, 150)
            }
        }
        return color_schemes.get(style, color_schemes['cute'])
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio (EAR) to detect blinks"""
        # Vertical distances
        v1 = self.calculate_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
        v2 = self.calculate_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
        # Horizontal distance
        h = self.calculate_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def calculate_mouth_aspect_ratio(self, landmarks):
        """Calculate Mouth Aspect Ratio (MAR) to detect mouth opening"""
        # Upper and lower lip landmarks
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        vertical = self.calculate_distance(upper_lip, lower_lip)
        horizontal = self.calculate_distance(left_mouth, right_mouth)
        
        mar = vertical / horizontal if horizontal > 0 else 0
        return mar
    
    def estimate_head_pose(self, landmarks, img_w, img_h):
        """Estimate head pose angles"""
        # Key facial landmarks for pose estimation
        nose_tip = landmarks[4]
        chin = landmarks[152]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        # Convert to pixel coordinates
        nose_2d = (nose_tip.x * img_w, nose_tip.y * img_h)
        chin_2d = (chin.x * img_w, chin.y * img_h)
        left_eye_2d = (left_eye.x * img_w, left_eye.y * img_h)
        right_eye_2d = (right_eye.x * img_w, right_eye.y * img_h)
        
        # Calculate yaw (left-right rotation)
        eye_center_x = (left_eye_2d[0] + right_eye_2d[0]) / 2
        face_center_x = img_w / 2
        yaw = (eye_center_x - face_center_x) / (img_w / 2) * 45
        
        # Calculate pitch (up-down rotation)
        nose_chin_distance = nose_2d[1] - chin_2d[1]
        pitch = (nose_chin_distance / img_h) * 90
        
        # Calculate roll (tilt)
        dy = right_eye_2d[1] - left_eye_2d[1]
        dx = right_eye_2d[0] - left_eye_2d[0]
        roll = math.degrees(math.atan2(dy, dx))
        
        return pitch, yaw, roll
    
    def detect_emotion(self, landmarks):
        """Simple emotion detection based on facial features"""
        # Calculate smile (mouth corners vs center)
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]
        
        # Mouth width vs height ratio
        mouth_width = self.calculate_distance(left_mouth, right_mouth)
        mouth_height = self.calculate_distance(mouth_top, mouth_bottom)
        smile_ratio = mouth_width / mouth_height if mouth_height > 0 else 0
        
        # Eyebrow position (simple check)
        left_brow = landmarks[70]
        right_brow = landmarks[300]
        nose_bridge = landmarks[6]
        
        brow_height = ((left_brow.y + right_brow.y) / 2 - nose_bridge.y)
        
        # Determine emotion
        if smile_ratio > 6.5:
            return 'happy'
        elif self.mouth_open_ratio > 0.6:
            return 'surprised'
        elif brow_height < -0.02:
            return 'angry'
        elif self.eye_open_ratio[0] < 0.3 and self.eye_open_ratio[1] < 0.3:
            return 'sleepy'
        else:
            return 'neutral'
    
    def process_hands(self, frame):
        """Process hand tracking"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Determine left or right hand
                hand_label = handedness.classification[0].label.lower()
                
                # Get hand position (wrist as reference)
                wrist = hand_landmarks.landmark[0]
                h, w = frame.shape[:2]
                hand_x = int(wrist.x * w)
                hand_y = int(wrist.y * h)
                
                self.hand_positions[hand_label] = (hand_x, hand_y)
                
                # Detect gesture
                self.hand_gestures[hand_label] = self.detect_hand_gesture(hand_landmarks)
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        else:
            self.hand_positions = {'left': None, 'right': None}
            self.hand_gestures = {'left': 'none', 'right': 'none'}
    
    def detect_hand_gesture(self, hand_landmarks):
        """Detect simple hand gestures"""
        # Get fingertip and base landmarks
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        index_base = hand_landmarks.landmark[5]
        middle_base = hand_landmarks.landmark[9]
        ring_base = hand_landmarks.landmark[13]
        pinky_base = hand_landmarks.landmark[17]
        
        # Check if fingers are extended
        index_extended = index_tip.y < index_base.y
        middle_extended = middle_tip.y < middle_base.y
        ring_extended = ring_tip.y < ring_base.y
        pinky_extended = pinky_tip.y < pinky_base.y
        
        # Peace sign (index and middle extended)
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return 'peace'
        # Open hand (all extended)
        elif index_extended and middle_extended and ring_extended and pinky_extended:
            return 'open'
        # Fist (none extended)
        elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'fist'
        # Pointing (only index)
        elif index_extended and not middle_extended:
            return 'point'
        else:
            return 'none'
    
    def apply_background_removal(self, frame):
        """Remove background and replace with custom color"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.selfie_segmentation.process(rgb_frame)
        
        # Create mask
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
        
        # Create background
        bg_image = np.zeros(frame.shape, dtype=np.uint8)
        bg_image[:] = self.background_color
        
        # Combine foreground and background
        output_image = np.where(condition, frame, bg_image)
        
        return output_image
    
    def start_recording(self, canvas_shape):
        """Start video recording"""
        if not self.is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vtuber_recording_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, 
                                                (canvas_shape[1], canvas_shape[0]))
            self.is_recording = True
            self.recording_start_time = time.time()
            return filename
        return None
    
    def stop_recording(self):
        """Stop video recording"""
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            self.is_recording = False
            self.video_writer = None
            self.recording_start_time = None
    
    def draw_avatar(self, canvas, show_landmarks=False):
        """Draw the 2D avatar based on tracked parameters"""
        center_x, center_y = self.avatar_center
        pitch, yaw, roll = self.head_rotation
        colors = self.avatar_colors
        
        # Draw head (circle that follows rotation)
        head_radius = 120
        head_offset_x = int(yaw * 2)
        head_offset_y = int(pitch * 2)
        
        # Head (circle with gradient effect)
        cv2.circle(canvas, (center_x + head_offset_x, center_y + head_offset_y), 
                   head_radius, colors['skin'], -1)
        cv2.circle(canvas, (center_x + head_offset_x, center_y + head_offset_y), 
                   head_radius, colors['skin_outline'], 3)
        
        # Blush marks (based on emotion)
        if self.emotion in ['happy', 'surprised']:
            blush_left_x = center_x + head_offset_x - 70
            blush_right_x = center_x + head_offset_x + 70
            blush_y = center_y + head_offset_y + 10
            cv2.circle(canvas, (blush_left_x, blush_y), 15, colors['blush'], -1)
            cv2.circle(canvas, (blush_right_x, blush_y), 15, colors['blush'], -1)
        
        # Eyes
        eye_y = center_y + head_offset_y - 20
        left_eye_x = center_x + head_offset_x - 40 + int(yaw)
        right_eye_x = center_x + head_offset_x + 40 + int(yaw)
        
        # Adjust eye size based on emotion
        eye_size_multiplier = 1.2 if self.emotion == 'surprised' else 1.0
        
        # Left eye
        eye_size_left = int(25 * self.eye_open_ratio[0] * eye_size_multiplier)
        if eye_size_left > 5:
            cv2.ellipse(canvas, (left_eye_x, eye_y), (22, eye_size_left), 
                       0, 0, 360, colors['eye_white'], -1)
            cv2.ellipse(canvas, (left_eye_x, eye_y), (22, eye_size_left), 
                       0, 0, 360, colors['eye_outline'], 2)
            cv2.circle(canvas, (left_eye_x + int(yaw/2), eye_y + int(pitch/2)), 
                      14, colors['iris'], -1)
            cv2.circle(canvas, (left_eye_x + int(yaw/2), eye_y + int(pitch/2)), 
                      8, colors['pupil'], -1)
            cv2.circle(canvas, (left_eye_x + int(yaw/2) + 3, eye_y + int(pitch/2) - 3), 
                      4, (255, 255, 255), -1)
        else:
            # Closed eye (line)
            cv2.line(canvas, (left_eye_x - 20, eye_y), 
                    (left_eye_x + 20, eye_y), colors['eye_outline'], 3)
        
        # Right eye
        eye_size_right = int(25 * self.eye_open_ratio[1] * eye_size_multiplier)
        if eye_size_right > 5:
            cv2.ellipse(canvas, (right_eye_x, eye_y), (22, eye_size_right), 
                       0, 0, 360, colors['eye_white'], -1)
            cv2.ellipse(canvas, (right_eye_x, eye_y), (22, eye_size_right), 
                       0, 0, 360, colors['eye_outline'], 2)
            cv2.circle(canvas, (right_eye_x + int(yaw/2), eye_y + int(pitch/2)), 
                      14, colors['iris'], -1)
            cv2.circle(canvas, (right_eye_x + int(yaw/2), eye_y + int(pitch/2)), 
                      8, colors['pupil'], -1)
            cv2.circle(canvas, (right_eye_x + int(yaw/2) + 3, eye_y + int(pitch/2) - 3), 
                      4, (255, 255, 255), -1)
        else:
            # Closed eye (line)
            cv2.line(canvas, (right_eye_x - 20, eye_y), 
                    (right_eye_x + 20, eye_y), colors['eye_outline'], 3)
        
        # Eyebrows (adjusted based on emotion)
        left_brow_y = eye_y - 35
        right_brow_y = eye_y - 35
        brow_angle = 0
        
        if self.emotion == 'angry':
            left_brow_y -= 5
            right_brow_y -= 5
            brow_angle = -15
        elif self.emotion == 'surprised':
            left_brow_y -= 10
            right_brow_y -= 10
        
        # Left eyebrow
        cv2.ellipse(canvas, (left_eye_x, left_brow_y), (25, 8), 
                   brow_angle, 0, 180, (80, 60, 40), 4)
        # Right eyebrow
        cv2.ellipse(canvas, (right_eye_x, right_brow_y), (25, 8), 
                   -brow_angle, 0, 180, (80, 60, 40), 4)
        
        # Mouth (adjusted based on emotion)
        mouth_y = center_y + head_offset_y + 40
        mouth_x = center_x + head_offset_x
        mouth_width = 50
        mouth_height = int(20 * self.mouth_open_ratio) if self.mouth_open_ratio > 0.3 else 5
        
        if self.emotion == 'happy':
            # Big smile
            cv2.ellipse(canvas, (mouth_x, mouth_y - 10), (mouth_width, 20), 
                       0, 0, 180, colors['mouth'], 4)
        elif mouth_height > 10:
            # Open mouth (ellipse)
            cv2.ellipse(canvas, (mouth_x, mouth_y), (mouth_width, mouth_height), 
                       0, 0, 360, colors['mouth'], -1)
            cv2.ellipse(canvas, (mouth_x, mouth_y), (mouth_width, mouth_height), 
                       0, 0, 360, (100, 50, 50), 2)
        else:
            # Closed mouth (smile)
            cv2.ellipse(canvas, (mouth_x, mouth_y - 5), (mouth_width, 15), 
                       0, 0, 180, colors['mouth'], 3)
        
        # Nose (simple triangle)
        nose_y = center_y + head_offset_y + 5
        nose_x = center_x + head_offset_x + int(yaw/2)
        nose_points = np.array([
            [nose_x, nose_y],
            [nose_x - 5, nose_y + 15],
            [nose_x + 5, nose_y + 15]
        ], np.int32)
        cv2.polylines(canvas, [nose_points], True, (180, 140, 110), 2)
        
        # Draw hands
        self.draw_hand_indicators(canvas)
        
        # Display info
        if show_landmarks:
            self.draw_debug_info(canvas)
    
    def draw_hand_indicators(self, canvas):
        """Draw hand indicators and gestures"""
        # Draw hands if detected
        for hand_type, position in self.hand_positions.items():
            if position:
                # Scale hand position to avatar canvas
                hand_canvas_x = 1000 + int((position[0] - 320) * 0.5)
                hand_canvas_y = 360 + int((position[1] - 240) * 0.5)
                
                gesture = self.hand_gestures[hand_type]
                
                # Draw hand indicator
                hand_color = (100, 200, 255) if hand_type == 'left' else (255, 200, 100)
                cv2.circle(canvas, (hand_canvas_x, hand_canvas_y), 30, hand_color, -1)
                cv2.circle(canvas, (hand_canvas_x, hand_canvas_y), 30, (255, 255, 255), 2)
                
                # Draw gesture icon
                if gesture == 'peace':
                    cv2.putText(canvas, 'V', (hand_canvas_x - 10, hand_canvas_y + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                elif gesture == 'open':
                    cv2.putText(canvas, '*', (hand_canvas_x - 10, hand_canvas_y + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                elif gesture == 'fist':
                    cv2.circle(canvas, (hand_canvas_x, hand_canvas_y), 15, (255, 255, 255), -1)
                elif gesture == 'point':
                    cv2.putText(canvas, '!', (hand_canvas_x - 8, hand_canvas_y + 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    def draw_debug_info(self, canvas):
        """Draw debug information overlay"""
        pitch, yaw, roll = self.head_rotation
        info_y = 30
        cv2.putText(canvas, f"Pitch: {pitch:.1f}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"Yaw: {yaw:.1f}", (10, info_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"Roll: {roll:.1f}", (10, info_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"Mouth: {self.mouth_open_ratio:.2f}", (10, info_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"Emotion: {self.emotion}", (10, info_y + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"FPS: {self.current_fps:.1f}", (10, info_y + 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def process_frame(self, frame):
        """Process video frame and update avatar parameters"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            
            h, w = frame.shape[:2]
            
            # Update head pose
            self.head_rotation = self.estimate_head_pose(landmarks, w, h)
            
            # Update eye state
            # Left eye landmarks: [33, 160, 158, 133, 153, 144]
            # Right eye landmarks: [362, 385, 387, 263, 373, 380]
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            
            left_ear = self.calculate_eye_aspect_ratio(landmarks, left_eye_indices)
            right_ear = self.calculate_eye_aspect_ratio(landmarks, right_eye_indices)
            
            # Normalize EAR to 0-1 range (typical EAR is around 0.2-0.4)
            self.eye_open_ratio[0] = min(1.0, max(0.0, left_ear * 3.5))
            self.eye_open_ratio[1] = min(1.0, max(0.0, right_ear * 3.5))
            
            # Update mouth state
            mouth_ratio = self.calculate_mouth_aspect_ratio(landmarks)
            self.mouth_open_ratio = min(1.0, max(0.0, mouth_ratio * 5))
            
            # Detect emotion
            self.emotion = self.detect_emotion(landmarks)
            
            # Draw face mesh on original frame (optional)
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            return True
        
        return False
    
    def run(self):
        """Main loop for VTuber application"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("=" * 60)
        print("VTuber Advanced - Controls:")
        print("Q: Quit")
        print("S: Toggle face mesh display")
        print("H: Toggle hand tracking")
        print("B: Toggle background removal")
        print("R: Start/Stop recording")
        print("1-4: Change avatar style (1=Cute, 2=Anime, 3=Cool, 4=Warm)")
        print("=" * 60)
        
        show_mesh = True
        show_hands = True
        
        while True:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Apply background removal if enabled
            if self.use_background_removal:
                frame = self.apply_background_removal(frame)
            
            # Process hands if enabled
            if show_hands:
                self.process_hands(frame)
            
            # Create canvas for avatar
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
            canvas[:, :] = (40, 40, 60)  # Dark blue-gray background
            
            # Process frame for face tracking
            face_detected = self.process_frame(frame)
            
            # Draw avatar
            self.draw_avatar(canvas, show_landmarks=True)
            
            # Prepare webcam feed
            if not show_mesh:
                # Show clean webcam without face mesh
                display_frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                clean_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                small_frame = cv2.resize(clean_frame, (320, 240))
            else:
                small_frame = cv2.resize(frame, (320, 240))
            
            # Place webcam feed on canvas
            canvas[10:250, 10:330] = small_frame
            cv2.rectangle(canvas, (10, 10), (330, 250), (255, 255, 255), 2)
            
            # Status indicators
            status_y = 30
            
            # Face detection status
            status_color = (0, 255, 0) if face_detected else (0, 0, 255)
            status_text = "Face Detected" if face_detected else "No Face"
            cv2.circle(canvas, (350, status_y), 10, status_color, -1)
            cv2.putText(canvas, status_text, (370, status_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Hand tracking status
            if show_hands:
                hands_detected = any(pos is not None for pos in self.hand_positions.values())
                hands_color = (0, 255, 0) if hands_detected else (100, 100, 100)
                cv2.circle(canvas, (350, status_y + 30), 10, hands_color, -1)
                cv2.putText(canvas, f"Hands: {show_hands}", (370, status_y + 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Background removal status
            bg_text = "BG Remove: ON" if self.use_background_removal else "BG Remove: OFF"
            bg_color = (0, 255, 0) if self.use_background_removal else (100, 100, 100)
            cv2.circle(canvas, (350, status_y + 60), 10, bg_color, -1)
            cv2.putText(canvas, bg_text, (370, status_y + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Recording status
            if self.is_recording:
                rec_time = int(time.time() - self.recording_start_time)
                rec_text = f"REC {rec_time}s"
                cv2.circle(canvas, (350, status_y + 90), 10, (0, 0, 255), -1)
                cv2.putText(canvas, rec_text, (370, status_y + 95), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                # Write frame to video
                self.video_writer.write(canvas)
            
            # Avatar style indicator
            cv2.putText(canvas, f"Style: {self.avatar_style.title()}", (350, status_y + 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Instructions
            cv2.putText(canvas, "Q:Quit | S:Mesh | H:Hands | B:BG | R:Record | 1-4:Style", 
                       (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Calculate and display FPS
            frame_time = time.time() - frame_start
            self.fps_counter.append(1.0 / frame_time if frame_time > 0 else 0)
            self.current_fps = sum(self.fps_counter) / len(self.fps_counter)
            
            # Show result
            cv2.imshow('VTuber Avatar Advanced', canvas)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_mesh = not show_mesh
                print(f"Face mesh: {'ON' if show_mesh else 'OFF'}")
            elif key == ord('h'):
                show_hands = not show_hands
                print(f"Hand tracking: {'ON' if show_hands else 'OFF'}")
            elif key == ord('b'):
                self.use_background_removal = not self.use_background_removal
                print(f"Background removal: {'ON' if self.use_background_removal else 'OFF'}")
            elif key == ord('r'):
                if not self.is_recording:
                    filename = self.start_recording(canvas.shape)
                    print(f"Recording started: {filename}")
                else:
                    self.stop_recording()
                    print("Recording stopped")
            elif key == ord('1'):
                self.avatar_style = 'cute'
                self.avatar_colors = self.get_avatar_colors('cute')
                print("Avatar style: Cute")
            elif key == ord('2'):
                self.avatar_style = 'anime'
                self.avatar_colors = self.get_avatar_colors('anime')
                print("Avatar style: Anime")
            elif key == ord('3'):
                self.avatar_style = 'cool'
                self.avatar_colors = self.get_avatar_colors('cool')
                print("Avatar style: Cool")
            elif key == ord('4'):
                self.avatar_style = 'warm'
                self.avatar_colors = self.get_avatar_colors('warm')
                print("Avatar style: Warm")
        
        # Cleanup
        if self.is_recording:
            self.stop_recording()
        cap.release()
        cv2.destroyAllWindows()

def main():
    print("=" * 70)
    print(" " * 20 + "VTuber Avatar Advanced")
    print(" " * 15 + "Pengolahan Citra Video Project")
    print("=" * 70)
    print("\nFeatures:")
    print("  - Real-time face tracking with emotion detection")
    print("  - Hand gesture recognition")
    print("  - Background removal (virtual green screen)")
    print("  - Video recording capability")
    print("  - Multiple avatar styles")
    print("  - Customizable colors and emotions")
    print("\nSelect Avatar Style:")
    print("  1. Cute (Pink & Warm)")
    print("  2. Anime (Blue & Vibrant)")
    print("  3. Cool (Blue & Cold tones)")
    print("  4. Warm (Orange & Earth tones)")
    print("\nEnter style number (1-4) [default: 1]: ", end="")
    
    try:
        choice = input().strip()
        if choice == '2':
            style = 'anime'
        elif choice == '3':
            style = 'cool'
        elif choice == '4':
            style = 'warm'
        else:
            style = 'cute'
    except:
        style = 'cute'
    
    print(f"\nInitializing VTuber with '{style.title()}' style...")
    print("=" * 70)
    
    vtuber = VTuberAvatar(avatar_style=style)
    vtuber.run()
    
    print("\nThank you for using VTuber Avatar!")
    print("Recording files saved in current directory.")

if __name__ == "__main__":
    main()
