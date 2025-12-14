# VTuber Avatar Configuration
# Customize your VTuber settings here

# ==================== AVATAR SETTINGS ====================

# Avatar position on canvas (x, y)
AVATAR_CENTER = (1000, 360)

# Avatar head radius (larger = bigger head)
HEAD_RADIUS = 120

# Avatar styles available: 'cute', 'anime', 'cool', 'warm'
DEFAULT_AVATAR_STYLE = 'cute'

# ==================== CANVAS SETTINGS ====================

# Canvas resolution (width, height)
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720

# Canvas background color (B, G, R)
CANVAS_BG_COLOR = (40, 40, 60)

# ==================== WEBCAM SETTINGS ====================

# Webcam index (usually 0 for default camera)
WEBCAM_INDEX = 0

# Webcam preview size on canvas (width, height)
WEBCAM_PREVIEW_SIZE = (320, 240)

# Webcam preview position (x, y)
WEBCAM_PREVIEW_POS = (10, 10)

# ==================== DETECTION SETTINGS ====================

# MediaPipe Face Mesh settings
FACE_DETECTION_CONFIDENCE = 0.5
FACE_TRACKING_CONFIDENCE = 0.5

# MediaPipe Hands settings
HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2

# Eye Aspect Ratio (EAR) multiplier for blink sensitivity
# Higher = eyes stay more open, Lower = more sensitive to blinks
EAR_MULTIPLIER = 3.5

# Mouth Aspect Ratio (MAR) multiplier for mouth opening sensitivity
# Higher = requires bigger mouth opening, Lower = more sensitive
MAR_MULTIPLIER = 5.0

# ==================== EMOTION DETECTION ====================

# Smile ratio threshold (higher = requires bigger smile)
SMILE_THRESHOLD = 6.5

# Surprised mouth ratio threshold
SURPRISED_THRESHOLD = 0.6

# Angry eyebrow height threshold
ANGRY_THRESHOLD = -0.02

# Sleepy eye aspect ratio threshold
SLEEPY_THRESHOLD = 0.3

# ==================== RECORDING SETTINGS ====================

# Recording codec (mp4v, XVID, etc.)
RECORDING_CODEC = 'mp4v'

# Recording FPS
RECORDING_FPS = 20.0

# Recording filename format
# Uses datetime.strftime format
RECORDING_FILENAME_FORMAT = "vtuber_recording_%Y%m%d_%H%M%S.mp4"

# ==================== BACKGROUND REMOVAL ====================

# Default background color when using background removal (B, G, R)
DEFAULT_BG_COLOR = (50, 150, 50)  # Green

# Selfie segmentation model (0 = general, 1 = landscape)
SEGMENTATION_MODEL = 1

# ==================== PERFORMANCE SETTINGS ====================

# FPS counter history length (smoothing)
FPS_HISTORY_LENGTH = 30

# ==================== UI SETTINGS ====================

# Show debug information by default
SHOW_DEBUG_INFO = True

# Info text color (B, G, R)
INFO_TEXT_COLOR = (255, 255, 255)

# Info text font scale
INFO_TEXT_SCALE = 0.6

# Info text thickness
INFO_TEXT_THICKNESS = 2

# Status indicator colors
STATUS_ACTIVE_COLOR = (0, 255, 0)  # Green
STATUS_INACTIVE_COLOR = (0, 0, 255)  # Red
STATUS_DISABLED_COLOR = (100, 100, 100)  # Gray

# ==================== ADVANCED SETTINGS ====================

# Enable/disable features by default
DEFAULT_HAND_TRACKING = True
DEFAULT_BACKGROUND_REMOVAL = False
DEFAULT_SHOW_MESH = True

# Hand gesture smoothing (not implemented yet)
GESTURE_SMOOTHING = True

# Avatar animation smoothing (not implemented yet)
ANIMATION_SMOOTHING = True

# ==================== CUSTOM COLORS ====================

# You can override avatar color schemes here
# Format: (B, G, R)

CUSTOM_AVATAR_COLORS = {
    'custom': {
        'skin': (255, 220, 180),
        'skin_outline': (200, 170, 140),
        'eye_white': (255, 255, 255),
        'eye_outline': (100, 100, 100),
        'iris': (100, 150, 200),
        'pupil': (50, 50, 50),
        'mouth': (255, 150, 150),
        'blush': (255, 180, 200)
    }
}

# ==================== KEYBOARD SHORTCUTS ====================

# You can customize these in the future
KEY_QUIT = 'q'
KEY_TOGGLE_MESH = 's'
KEY_TOGGLE_HANDS = 'h'
KEY_TOGGLE_BG = 'b'
KEY_TOGGLE_RECORD = 'r'
KEY_STYLE_1 = '1'
KEY_STYLE_2 = '2'
KEY_STYLE_3 = '3'
KEY_STYLE_4 = '4'
