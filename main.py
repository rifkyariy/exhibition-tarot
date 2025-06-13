import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import numpy as np
import requests
import json
import re
import time
import threading

from mediapipe.framework.formats import landmark_pb2

class HandGestureControlApp:
    """
    A class to encapsulate the hand gesture control application logic,
    including MediaPipe setup, OpenCV camera handling, custom gesture detection,
    and UI rendering.
    """

    # --- Configuration Constants ---
    MODEL_PATH = 'model/gesture_recognizer.task'
    PINCH_THRESHOLD = 0.045 # Threshold for custom pinch detection (normalized distance)
    
    # Base resolution for UI scaling - actual camera resolution might differ
    BASE_CAM_WIDTH = 1280
    BASE_CAM_HEIGHT = 720

    COLOR_PICKER_WIDTH_RATIO = 0.6 # 60% of frame width
    COLOR_PICKER_HEIGHT_RATIO = 0.4 # 40% of frame height

    GESTURE_TOGGLE_COOLDOWN_FRAMES = 20 # Cooldown for peace sign toggle
    POEM_GENERATION_COOLDOWN_FRAMES = 100 # Cooldown for poem generation (approx 3 seconds at 30 FPS)

    # --- UI Configuration Constants ---
    UI_TEXT_COLOR = (255, 255, 255)  # White
    UI_ACCENT_COLOR = (0, 200, 0)    # Greenish
    UI_ERROR_COLOR = (0, 0, 255)     # Red
    UI_PANEL_BG_COLOR = (30, 30, 30) # Dark Gray
    UI_PANEL_ALPHA = 0.6             # Transparency for UI panels
    UI_BORDER_COLOR = (60, 60, 60)   # Panel border color

    HEADER_FONT_SCALE = 0.0015 # Scaled to image height
    SUBHEADER_FONT_SCALE = 0.0011
    BODY_FONT_SCALE = 0.0008
    
    THICKNESS_NORMAL = 1
    THICKNESS_BOLD = 2
    PADDING_RATIO = 0.02 # Padding as a ratio of the smaller dimension (height or width)
    LINE_HEIGHT_FACTOR = 1.6 # Multiplier for font height to get line spacing

    def __init__(self):
        """Initializes the MediaPipe recognizer, OpenCV camera, and application state variables."""
        self.cap = None
        self.recognizer = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # Application state variables
        self.drag_start_pos = None
        self.current_drag_pos = None

        self.is_color_picker_active = False
        self.is_hand_picking_color = False
        self.was_hand_picking_color_in_prev_frame = False

        self.current_overlay_color = (0, 0, 255) # Initial color (BGR: Red)
        self.current_color_name = "Red"
        self.color_spectrum_image = None
        # Relative position within the picker (0-1) to preserve selection across resizes
        self.last_picked_relative_pos_on_picker = (0.5, 0.5) 

        self.current_gesture_cooldown = 0
        self.gesture_active_in_prev_frame = False
        self.is_peace_sign_detected_this_frame = False # Flag for current frame

        self.any_hand_detected_this_frame = False # New flag for hand presence

        self.total_frames = 0
        self.gesture_counts = {
            "Thumb Up": 0, "Thumb Down": 0, "Peace Sign": 0,
            "Index Up": 0, "Closed": 0, "Open": 0, "I Love You": 0,
            "Pinching": 0, "None": 0
        }
        self.frame_timestamp_ms = 0

        # Poem generation related variables
        self.generated_poem = "Move your hand to pick a color and generate a poem!"
        self.last_color_for_poem_gen = self.current_overlay_color
        self.poem_generation_cooldown = 0 # Cooldown for when a new poem can be requested
        self.mood_text = ""
        self.warmth_text = ""

        # New variables for non-blocking poem generation
        self.is_generating_poem = False # Flag to indicate if poem generation is in progress
        self.poem_thread = None # To hold the thread object

    def _get_font_scale(self, base_scale, img_height):
        """Calculates dynamic font scale based on image height."""
        return base_scale * img_height

    def _get_padding(self, img_width, img_height):
        """Calculates dynamic padding based on the smaller image dimension."""
        return int(min(img_width, img_height) * self.PADDING_RATIO)

    def _display_loading_screen(self, stage_name, progress, total_stages):
        """Displays a loading screen with a progress bar."""
        width, height = 800, 600 # Fixed size for loading screen as it's pre-camera
        loading_screen = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background
        cv2.rectangle(loading_screen, (0, 0), (width, height), (30, 30, 30), -1)

        # Dynamic font scales for loading screen
        header_fs = self._get_font_scale(self.HEADER_FONT_SCALE, height) * 1.5
        subheader_fs = self._get_font_scale(self.SUBHEADER_FONT_SCALE, height) * 1.5

        # Title
        title_text = "Loading Hand Gesture Control"
        (text_w, text_h), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, header_fs, self.THICKNESS_BOLD)
        cv2.putText(loading_screen, title_text, ((width - text_w) // 2, height // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, header_fs, self.UI_TEXT_COLOR, self.THICKNESS_BOLD, cv2.LINE_AA)

        # Stage name
        stage_text = f"Initializing: {stage_name}..."
        (stage_w, stage_h), _ = cv2.getTextSize(stage_text, cv2.FONT_HERSHEY_SIMPLEX, subheader_fs, self.THICKNESS_NORMAL)
        cv2.putText(loading_screen, stage_text, ((width - stage_w) // 2, height // 2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, subheader_fs, (180, 180, 180), self.THICKNESS_NORMAL, cv2.LINE_AA)

        # Progress bar
        bar_width = int(width * 0.6)
        bar_height = 20
        bar_x = (width - bar_width) // 2
        bar_y = height // 2

        # Draw empty bar
        cv2.rectangle(loading_screen, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 2)

        # Draw filled part
        filled_width = int(bar_width * (progress / total_stages))
        cv2.rectangle(loading_screen, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), self.UI_ACCENT_COLOR, -1)

        # Progress percentage text
        percent_text = f"{int((progress / total_stages) * 100)}%"
        (percent_w, percent_h), _ = cv2.getTextSize(percent_text, cv2.FONT_HERSHEY_SIMPLEX, self._get_font_scale(self.BODY_FONT_SCALE, height) * 1.5, self.THICKNESS_NORMAL)
        cv2.putText(loading_screen, percent_text, (bar_x + filled_width - percent_w - 5, bar_y + bar_height - (bar_height - percent_h) // 2 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, self._get_font_scale(self.BODY_FONT_SCALE, height) * 1.5, self.UI_TEXT_COLOR, self.THICKNESS_NORMAL, cv2.LINE_AA)

        cv2.imshow('Hand Gesture Control', loading_screen)
        cv2.waitKey(1) # Refresh the window

    def initialize_app(self):
        """Initializes components with a loading progress bar."""
        total_stages = 3
        current_stage = 0

        cv2.namedWindow('Hand Gesture Control', cv2.WINDOW_AUTOSIZE)

        # Stage 1: Initializing MediaPipe
        current_stage += 1
        self._display_loading_screen("MediaPipe", current_stage, total_stages)
        time.sleep(0.5) # Simulate work
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        # Stage 2: Initializing Camera
        current_stage += 1
        self._display_loading_screen("Camera", current_stage, total_stages)
        time.sleep(0.5) # Simulate work
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()
        
        # Set desired camera resolution (might not be exactly achieved)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.BASE_CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.BASE_CAM_HEIGHT)
        
        # Stage 3: Preparing UI Elements
        current_stage += 1
        self._display_loading_screen("UI Elements", current_stage, total_stages)
        time.sleep(0.5) # Simulate work
        # Pre-generate color spectrum image
        # This will be scaled later when drawn
        self.color_spectrum_image = self._generate_custom_gradient_image(
            int(self.BASE_CAM_WIDTH * self.COLOR_PICKER_WIDTH_RATIO),
            int(self.BASE_CAM_HEIGHT * self.COLOR_PICKER_HEIGHT_RATIO)
        )

        # Final display of 100% before main loop starts
        self._display_loading_screen("Ready!", total_stages, total_stages)
        time.sleep(0.5)

    @staticmethod
    def _calculate_distance(p1, p2):
        """Calculates the Euclidean distance between two MediaPipe NormalizedLandmark points."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    @staticmethod
    def _generate_custom_gradient_image(width, height):
        """
        Generates a 2D color gradient image for the color picker,
        transitioning horizontally from cool (blue) to warm (orange),
        and vertically controlling brightness.
        """
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Define start (cool) and end (warm) colors in BGR format for the horizontal gradient
        start_hue_color_bgr = np.array([255, 0, 0], dtype=np.float32)   # Blue (cool)
        end_hue_color_bgr = np.array([0, 165, 255], dtype=np.float32) # Orange (warm) - BGR: Blue=0, Green=165, Red=255

        for x in range(width):
            # Calculate horizontal interpolation for the base hue color (cool to warm)
            factor_x = x / (width - 1)
            base_color = start_hue_color_bgr * (1 - factor_x) + end_hue_color_bgr * factor_x

            for y in range(height):
                # Calculate vertical interpolation for brightness
                if y < height / 2:
                    # Top half: Blend base_color towards white (brighter)
                    alpha_white = 1.0 - (y / (height / 2.0)) if height > 0 else 0.0
                    final_color = base_color * (1.0 - alpha_white) + np.array([255, 255, 255], dtype=np.float32) * alpha_white
                else:
                    # Bottom half: Blend base_color towards black (darker)
                    alpha_black = (y - height / 2.0) / (height / 2.0) if height > 0 else 0.0
                    final_color = base_color * (1.0 - alpha_black) + np.array([0, 0, 0], dtype=np.float32) * alpha_black

                image[y, x] = np.clip(final_color, 0, 255).astype(np.uint8)
        return image

    def _process_hand_landmarks(self, image, recognition_result):
        """
        Processes detected hand landmarks, performs custom gesture detection,
        and handles color picker interactions.
        """
        self.any_hand_pinching_this_frame = False
        self.is_hand_picking_color = False # Reset for current frame calculation
        self.any_hand_detected_this_frame = bool(recognition_result.hand_landmarks) # Update hand presence flag

        # Reset current frame's specific gesture counts for accurate percentage calculation
        temp_gesture_counts_this_frame = {key: 0 for key in self.gesture_counts}

        for hand_index, hand_landmarks_list in enumerate(recognition_result.hand_landmarks):
            # Convert list of NormalizedLandmark to NormalizedLandmarkList protobuf for drawing
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            for landmark in hand_landmarks_list:
                hand_landmarks_proto.landmark.add(x=landmark.x, y=landmark.y, z=landmark.z)

            self.mp_drawing.draw_landmarks(image, hand_landmarks_proto, self.mp_hands.HAND_CONNECTIONS)

            current_hand_gestures = recognition_result.gestures[hand_index]
            current_hand_gesture_name = "None"
            is_current_hand_pinching = False

            # --- Custom Pinching Detection Logic ---
            # Landmarks for thumb tip (4) and index finger tip (8)
            thumb_tip = hand_landmarks_list[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks_list[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinch_distance = self._calculate_distance(thumb_tip, index_tip)

            if pinch_distance < self.PINCH_THRESHOLD:
                is_current_hand_pinching = True
                current_hand_gesture_name = "Pinching" # Prioritize custom pinch

            # Process MediaPipe model's output if custom pinch wasn't detected
            if not is_current_hand_pinching and current_hand_gestures:
                detected_gesture_name_from_model = current_hand_gestures[0].category_name
                # Map model output names to our desired display names
                if detected_gesture_name_from_model == "Closed_Fist":
                    current_hand_gesture_name = "Closed"
                elif detected_gesture_name_from_model == "Open_Palm":
                    current_hand_gesture_name = "Open"
                elif detected_gesture_name_from_model == "Pointing_Up":
                    current_hand_gesture_name = "Index Up"
                elif detected_gesture_name_from_model == "Thumb_Up":
                    current_hand_gesture_name = "Thumb Up"
                elif detected_gesture_name_from_model == "Thumb_Down":
                    current_hand_gesture_name = "Thumb Down"
                elif detected_gesture_name_from_model == "Victory":
                    current_hand_gesture_name = "Peace Sign"
                elif detected_gesture_name_from_model == "ILoveYou":
                    current_hand_gesture_name = "I Love You"
                elif detected_gesture_name_from_model == "Pinching": # For custom trained models
                    current_hand_gesture_name = "Pinching"
                else:
                    current_hand_gesture_name = "None"
            elif not is_current_hand_pinching and not current_hand_gestures:
                current_hand_gesture_name = "None"

            # Set global flag for Peace Sign toggle
            if current_hand_gesture_name == "Peace Sign":
                self.is_peace_sign_detected_this_frame = True

            # Increment temporary count for this specific gesture
            temp_gesture_counts_this_frame[current_hand_gesture_name] += 1

            # Update drag positions if current hand is pinching
            if is_current_hand_pinching:
                self.any_hand_pinching_this_frame = True
                self.current_drag_pos = (
                    int(hand_landmarks_list[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]),
                    int(hand_landmarks_list[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0])
                )

                # Hand control for color picker (Pinch & Drag)
                if self.is_color_picker_active and self.current_drag_pos:
                    picker_x1, picker_y1, picker_width, picker_height = self._get_color_picker_coords(image)
                    
                    # Convert drag position to relative position on the picker
                    relative_x_on_picker = (self.current_drag_pos[0] - picker_x1) / picker_width
                    relative_y_on_picker = (self.current_drag_pos[1] - picker_y1) / picker_height

                    # Clamp to [0, 1] range
                    relative_x_on_picker = np.clip(relative_x_on_picker, 0, 0.999)
                    relative_y_on_picker = np.clip(relative_y_on_picker, 0, 0.999)

                    self.last_picked_relative_pos_on_picker = (relative_x_on_picker, relative_y_on_picker)


                    # Get color from the pre-generated spectrum image based on relative position
                    # Need to scale the pre-generated image's width/height to current picker size
                    scaled_spectrum_width = self.color_spectrum_image.shape[1]
                    scaled_spectrum_height = self.color_spectrum_image.shape[0]

                    color_x_in_spectrum = int(relative_x_on_picker * scaled_spectrum_width)
                    color_y_in_spectrum = int(relative_y_on_picker * scaled_spectrum_height)
                    
                    self.current_overlay_color = tuple(self.color_spectrum_image[color_y_in_spectrum, color_x_in_spectrum].tolist())
                    self.current_color_name = "Custom"
                    self.is_hand_picking_color = True

                if self.drag_start_pos is None:
                    self.drag_start_pos = self.current_drag_pos
            else:
                pass # This hand is not pinching

            # Visual feedback: Draw a circle at the pinch point if pinching
            if is_current_hand_pinching and self.current_drag_pos:
                cv2.circle(image, self.current_drag_pos, 10, self.current_overlay_color, -1)

        # Update global gesture_counts
        for gesture_name, count_val in temp_gesture_counts_this_frame.items():
            if count_val > 0:
                self.gesture_counts[gesture_name] += 1
        
        # Reset drag state if no hand is pinching
        if not self.any_hand_pinching_this_frame:
            self.drag_start_pos = None

    def _get_color_picker_coords(self, image):
        """Calculates and returns the coordinates (x1, y1, width, height) of the color picker."""
        img_height, img_width, _ = image.shape
        
        picker_width = int(img_width * self.COLOR_PICKER_WIDTH_RATIO)
        picker_height = int(img_height * self.COLOR_PICKER_HEIGHT_RATIO)

        picker_x1 = (img_width - picker_width) // 2
        picker_y1 = img_height - picker_height - self._get_padding(img_width, img_height) * 2 # Padding from bottom
        
        return picker_x1, picker_y1, picker_width, picker_height

    def _generate_poem(self, mood, warmth):
        """Generates a short poem using the Ollama API based on mood and warmth."""
        # More specific prompt that discourages reasoning output
        prompt_text = f"""
            Write a poem (max 7 lines) inspired by a {mood.lower()} and {warmth.lower()}, focusing on themes of connection, fate, and the bittersweet nature of relationships.
            Based on the color {self.current_color_name}: make it emotionally lyrical, reminiscent of reflective, narrative song lyrics.
            Use vivid imagery and metaphors, weaving in elements of destiny, human interaction, and internal conflict.
            Respond only with the poem, no explanations or reasoning.
            The poem should evoke feelings and imagery related to the color and its mood, with a flowing, narrative style.
            Include:
            - A sense of two people drawn together, a fated or chosen connection.
            - Imagery of celestial bodies or grand, universal themes.
            - A mix of hopeful beginnings and a subtle undertone of potential difficulty or a turning point.
            - Dialogue or implied conversation between two individuals.
            - Reflections on fate, choice, and the passage of time.
            - Emotional weight and a sense of deep personal feeling.
            - A tone that is introspective, slightly melancholic, but with a lingering hope or acceptance.
            - Focus on the feelings evoked by the color and the emotional arc of the relationship, rather than describing the color directly.
            - A strong sense of devotion or admiration for another individual, elevating them to an almost mythical status.
            - Hyperbolic or cosmic imagery to express the depth of feeling (e.g., waiting on the moon, celestial references).
            - A recognition of potential pain or heartbreak alongside the deep connection.
            - A tone that is romantic, melancholic, wistful, and deeply introspective, like a confession or a heartfelt reflection.
            - Focus on the feeling the color evokes and the emotional arc of the relationship, rather than describing the color directly.
            - More warmth and emotional depth should signify happiness and strong, unyielding connection; less warmth, more melancholy or a sense of destined heartbreak.
            - Avoid using word "color" or directly referencing the color in the poem and "beneath" the poem.
            The poem should feel like a concise yet complete lyrical piece, capturing a significant emotional moment in a relationship.
            Avoid any reasoning or meta-commentary in your response. Focus solely on the poem itself.
            """
        
        try:
            # Ollama API endpoint for generating text
            ollama_api_url = "http://localhost:11434/api/generate" 
            
            # Updated payload with system prompt and temperature settings
            payload = {
                "model": "gemma3:latest", # Ensure this model is downloaded in Ollama (ollama pull gemma3:1b)
                "prompt": prompt_text,
                "stream": False,
                "options": {
                    "temperature": 0.9,   # Add some creativity
                    "top_p": 0.9,
                    "num_predict": 150    # Limit output length
                },
                "system": "You are a genius poet. Respond only with poems, no explanations or reasoning. Do not include any reasoning or meta-commentary in your response. Focus on the poem itself."
            }
            
            headers = {'Content-Type': 'application/json'}
            
            response = requests.post(ollama_api_url, headers=headers, data=json.dumps(payload), timeout=20) # Added timeout
            response.raise_for_status()
            
            result = response.json()

            if 'response' in result:
                poem_text = result['response'].strip()

                # Convert smart quotes to standard ASCII quotes
                poem_text = poem_text.replace('\u201c', '').replace('\u201d', '')
                poem_text = poem_text.replace('\u2018', "").replace('\u2019', "") 

                
                # Enhanced cleaning to remove reasoning artifacts
                lines = poem_text.split('\n')
                cleaned_lines = []
                
                # Skip lines that look like reasoning or meta-commentary
                skip_patterns = [
                    r'^(Alright|Let me|First|Next|Then|Finally|I want|I\'ll|Putting)',
                    r'^(Think|Looking|The user|This)',
                    r'^\s*<think>',
                    r'^\s*</think>',
                    r'step by step',
                    r'brainstorm',
                    r'understand the mood',
                    r'I need to',
                    r'As an AI language model',
                    r'Here is a short poem',
                    r'The poem below'
                ]
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        cleaned_lines.append('')   # Preserve empty lines for stanza breaks
                        continue
                        
                    # Check if line matches any skip pattern
                    should_skip = False
                    for pattern in skip_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            should_skip = True
                            break
                    
                    if not should_skip:
                        # Remove common prefixes
                        line = re.sub(r'^(?:Here is|Here\'s|A poem about):?\s*', '', line, flags=re.IGNORECASE)
                        if line:   # Only add non-empty lines after cleaning
                            cleaned_lines.append(line)
                
                # Join the cleaned lines and do final cleanup
                cleaned_poem = '\n'.join(cleaned_lines).strip()
                
                # Remove any remaining artifacts
                cleaned_poem = re.sub(r'^```.*?```$', '', cleaned_poem, flags=re.MULTILINE | re.DOTALL)
                cleaned_poem = re.sub(r'^\s*Generated Poem:?\s*', '', cleaned_poem, flags=re.IGNORECASE | re.MULTILINE)
                
                # If the poem is too long, truncate to approximately 12 lines
                final_lines = [line for line in cleaned_poem.split('\n') if line.strip()]
                if len(final_lines) > 12:
                    final_lines = final_lines[:12]
                
                return '\n'.join(final_lines) if final_lines else "A color speaks in silence,\nWhispers of light and shadow."
                
            else:
                print(f"Ollama API response did not contain expected text: {result}")
                return "Could not generate poem: Unexpected Ollama API response."
                
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to Ollama server. Please ensure Ollama is running at http://localhost:11434.")
            return "Could not generate poem: Ollama server not reachable. Ensure Ollama is running."
        except requests.exceptions.Timeout:
            print("Error: Ollama API request timed out.")
            return "Could not generate poem: Ollama API timed out."
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return "Could not generate poem: Network error or API issue."
        except json.JSONDecodeError:
            print("Error decoding JSON from Ollama API response.")
            return "Could not generate poem: Invalid Ollama API response."
        except Exception as e:
            print(f"An unexpected error occurred during poem generation: {e}")
            return "Could not generate poem: An unexpected error occurred."

    def _start_poem_generation_thread(self, mood, warmth):
        """Starts a new thread to generate the poem."""
        if not self.is_generating_poem and (self.poem_thread is None or not self.poem_thread.is_alive()):
            self.is_generating_poem = True
            self.generated_poem = "Generating poem..." # Display interim message
            self.poem_thread = threading.Thread(target=self._run_poem_generation_in_background, args=(mood, warmth))
            self.poem_thread.daemon = True # Allow the main program to exit even if thread is running
            self.poem_thread.start()

    def _run_poem_generation_in_background(self, mood, warmth):
        """Method executed in the background thread to generate the poem."""
        print(f"THREAD: Starting poem generation for Mood: {mood}, Warmth: {warmth}")
        generated_text = self._generate_poem(mood, warmth)
        self.generated_poem = generated_text
        self.is_generating_poem = False
        print("THREAD: Poem generation finished.")

    def _draw_transparent_rectangle(self, image, rect_coords, color, alpha):
        """Draws a semi-transparent rectangle on the image with a border."""
        overlay = image.copy()
        x1, y1, x2, y2 = rect_coords
        
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1: return # Invalid rectangle

        sub_overlay = overlay[y1:y2, x1:x2]
        cv2.rectangle(sub_overlay, (0, 0), (sub_overlay.shape[1], sub_overlay.shape[0]), color, -1)
        image[y1:y2, x1:x2] = cv2.addWeighted(sub_overlay, alpha, image[y1:y2, x1:x2], 1 - alpha, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), self.UI_BORDER_COLOR, 2) # Subtle border

    def _display_multiline_text(self, image, text, start_coords, font_scale, color, thickness, line_spacing_factor=1.2, max_width_ratio=0.4):
        """
        Displays multiline text on the OpenCV image, handling word wrapping.
        :param image: The OpenCV image to draw on.
        :param text: The text string to display.
        :param start_coords: A tuple (x, y) for the top-left corner of the text area.
        :param font_scale: Font scale for cv2.putText.
        :param color: Text color (BGR tuple).
        :param thickness: Line thickness for cv2.putText.
        :param line_spacing_factor: Factor to multiply font height by for line spacing.
        :param max_width_ratio: Max width of the text block as a ratio of image width.
        """
        img_width = image.shape[1]
        max_text_width = int(img_width * max_width_ratio)
        current_y = start_coords[1]
        x = start_coords[0]

        # Split into paragraphs/lines by newline characters
        paragraphs = text.split('\n')

        for para in paragraphs:
            words = para.split(' ')
            current_line = []
            for word in words:
                # Check if adding the next word exceeds max width
                test_line = ' '.join(current_line + [word])
                (text_w, text_h), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                if text_w > max_text_width and current_line:
                    # If current line exceeds max width, draw it and start a new line
                    cv2.putText(image, ' '.join(current_line), (x, current_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
                    current_y += int(text_h * line_spacing_factor)
                    current_line = [word] # Start new line with the current word
                else:
                    current_line.append(word)
            
            # Draw any remaining words in the current line
            if current_line:
                cv2.putText(image, ' '.join(current_line), (x, current_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
                current_y += int(text_h * line_spacing_factor)
            
            # Add extra space between paragraphs
            if para != paragraphs[-1]: # If not the last paragraph
                current_y += int(text_h * (line_spacing_factor / 2)) # Add half a line space

    def _draw_cooldown_bar(self, image, x, y, width, height, current_value, max_value, color=(0,255,0)):
        """Draws a horizontal progress bar."""
        cv2.rectangle(image, (x, y), (x + width, y + height), self.UI_PANEL_BG_COLOR, -1) # Background bar
        
        fill_width = int(width * (current_value / max_value))
        cv2.rectangle(image, (x, y), (x + fill_width, y + height), color, -1) # Filled portion
        cv2.rectangle(image, (x, y), (x + width, y + height), self.UI_TEXT_COLOR, 1) # Border

    def _draw_ui_elements(self, image):
        """Draws various UI elements on the image dynamically."""
        img_height, img_width, _ = image.shape
        padding = self._get_padding(img_width, img_height)

        # Dynamic font scales
        header_fs = self._get_font_scale(self.HEADER_FONT_SCALE, img_height)
        subheader_fs = self._get_font_scale(self.SUBHEADER_FONT_SCALE, img_height)
        body_fs = self._get_font_scale(self.BODY_FONT_SCALE, img_height)

        # --- Top Left Panel: Greeting & Color Info ---
        panel1_x1 = padding
        panel1_y1 = padding
        panel1_width = int(img_width * 0.25)
        panel1_height = int(img_height * 0.35)
        panel1_x2 = panel1_x1 + panel1_width
        panel1_y2 = panel1_y1 + panel1_height

        self._draw_transparent_rectangle(image, (panel1_x1, panel1_y1, panel1_x2, panel1_y2),
                                        self.UI_PANEL_BG_COLOR, self.UI_PANEL_ALPHA)

        # Greeting Title
        greeting_text = "How was your day?"
        (title1_w, title1_h), _ = cv2.getTextSize(greeting_text, cv2.FONT_HERSHEY_SIMPLEX, subheader_fs, self.THICKNESS_BOLD)
        current_y_pos = panel1_y1 + padding + title1_h
        cv2.putText(image, greeting_text, (panel1_x1 + padding, current_y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, subheader_fs, self.UI_ACCENT_COLOR, self.THICKNESS_BOLD, cv2.LINE_AA)

        current_y_pos += int(title1_h * self.LINE_HEIGHT_FACTOR * 0.5)  # Adjusted padding (removed color name)

        # Color Swatch Padding
        swatch_padding_y = int(padding * 0.3)
        current_y_pos += swatch_padding_y

        # Color Swatch
        color_swatch_size = int(img_height * 0.04)
        swatch_x1 = panel1_x1 + padding
        swatch_y1 = current_y_pos
        swatch_x2 = swatch_x1 + color_swatch_size
        swatch_y2 = swatch_y1 + color_swatch_size

        cv2.rectangle(image, (swatch_x1, swatch_y1), (swatch_x2, swatch_y2),
                    self.current_overlay_color, -1)
        cv2.rectangle(image, (swatch_x1, swatch_y1), (swatch_x2, swatch_y2),
                    self.UI_TEXT_COLOR, 1)

        current_y_pos = swatch_y2 + swatch_padding_y

        # Hex, Mood, Warmth
        b, g, r = self.current_overlay_color
        hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
        self.hex_color = hex_color  # Store hex color for poem generation
        
        color_bgr_np = np.uint8([[self.current_overlay_color]])
        color_hsv = cv2.cvtColor(color_bgr_np, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = color_hsv[0], color_hsv[1], color_hsv[2]

        mood_text = ""
        if s < 50: mood_text = "Muted"
        elif s < 150: mood_text = "Balanced"
        else: mood_text = "Vibrant"
        if v < 80: mood_text += " & Dark"
        elif v > 180: mood_text += " & Bright"
        else: mood_text += " & Normal"
        self.mood_text = mood_text

        warmth_diff = int(r) - int(b)
        warmth_text = ""
        if warmth_diff > 100: warmth_text = "Mood"
        elif warmth_diff > 40: warmth_text = "Warm"
        elif warmth_diff < -100: warmth_text = "Blue"
        elif warmth_diff < -40: warmth_text = "Cool"
        else: warmth_text = "Neutral"
        self.warmth_text = warmth_text

        cv2.putText(image, f"Hex: {hex_color}", (panel1_x1 + padding, current_y_pos + int(color_swatch_size * 0.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, body_fs, self.UI_TEXT_COLOR, self.THICKNESS_NORMAL, cv2.LINE_AA)
        current_y_pos += int(cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, body_fs, self.THICKNESS_NORMAL)[0][1] * self.LINE_HEIGHT_FACTOR)

        cv2.putText(image, f"Mood: {mood_text}", (panel1_x1 + padding, current_y_pos + int(color_swatch_size * 0.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, body_fs, self.UI_TEXT_COLOR, self.THICKNESS_NORMAL, cv2.LINE_AA)
        current_y_pos += int(cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, body_fs, self.THICKNESS_NORMAL)[0][1] * self.LINE_HEIGHT_FACTOR)

        cv2.putText(image, f"Warmth: {warmth_text}", (panel1_x1 + padding, current_y_pos + int(color_swatch_size * 0.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, body_fs, self.UI_TEXT_COLOR, self.THICKNESS_NORMAL, cv2.LINE_AA)

        # The rest of your original UI (poem panel, color picker, instructions) can follow this section unchanged



        # --- Hand Presence Indicator (Top Center) ---
        hand_status_text = "HAND: DETECTED" if self.any_hand_detected_this_frame else "HAND: NOT DETECTED"
        hand_status_color = self.UI_ACCENT_COLOR if self.any_hand_detected_this_frame else self.UI_ERROR_COLOR
        (hs_w, hs_h), _ = cv2.getTextSize(hand_status_text, cv2.FONT_HERSHEY_SIMPLEX, subheader_fs, self.THICKNESS_BOLD)
        # cv2.putText(image, hand_status_text, ((img_width - hs_w) // 2, padding + hs_h),
        #             cv2.FONT_HERSHEY_SIMPLEX, subheader_fs, hand_status_color, self.THICKNESS_BOLD, cv2.LINE_AA)


        # --- Top Right Panel: Poem Display ---
        poem_panel_width = int(img_width * 0.45) # About 45% of screen width
        poem_panel_height = int(img_height * 0.40) # About 40% of screen height
        poem_panel_x1 = img_width - poem_panel_width - padding
        poem_panel_y1 = padding
        poem_panel_x2 = poem_panel_x1 + poem_panel_width
        poem_panel_y2 = poem_panel_y1 + poem_panel_height

        self._draw_transparent_rectangle(image, (poem_panel_x1, poem_panel_y1, poem_panel_x2, poem_panel_y2),
                                         self.UI_PANEL_BG_COLOR, self.UI_PANEL_ALPHA)

        # Calculate precise Y for "GENERATED POEM" title
        (title2_w, title2_h), _ = cv2.getTextSize("GENERATED POEM", cv2.FONT_HERSHEY_SIMPLEX, subheader_fs, self.THICKNESS_BOLD)
        poem_title_y = poem_panel_y1 + padding + title2_h # Position title with padding from top, plus its height
        cv2.putText(image, "GENERATED POEM", (poem_panel_x1 + padding, poem_title_y),
                    cv2.FONT_HERSHEY_SIMPLEX, subheader_fs, self.UI_ACCENT_COLOR, self.THICKNESS_BOLD, cv2.LINE_AA)
        
        # Poem Text start position
        poem_text_start_y = poem_title_y + int(title2_h * self.LINE_HEIGHT_FACTOR) + padding // 2

        if self.is_generating_poem:
            dots = ""
            # Pulsing dots animation
            if self.total_frames % 45 < 15:
                dots = "."
            elif self.total_frames % 45 < 30:
                dots = ".."
            else:
                dots = "..."
            poem_display_text = f"Generating poem{dots}"
            
            # Simple pulsating color effect for text
            pulsate_val = abs(math.sin(time.time() * 5)) # oscillates between 0 and 1
            pulsate_color_component = int(200 + 55 * pulsate_val) # From 200 to 255
            pulsate_color = (pulsate_color_component, pulsate_color_component, pulsate_color_component)

            self._display_multiline_text(image, poem_display_text, (poem_panel_x1 + padding, poem_text_start_y),
                                         subheader_fs, pulsate_color, self.THICKNESS_BOLD,
                                         line_spacing_factor=1.5, max_width_ratio=0.4)
        else:
            self._display_multiline_text(image, self.generated_poem, (poem_panel_x1 + padding, poem_text_start_y),
                                         body_fs, self.UI_TEXT_COLOR, self.THICKNESS_NORMAL,
                                         line_spacing_factor=self.LINE_HEIGHT_FACTOR, max_width_ratio=0.4)

        # --- Color Picker Overlay if active ---
        if self.is_color_picker_active:
            picker_x1, picker_y1, picker_width, picker_height = self._get_color_picker_coords(image)
            
            # Draw semi-transparent dark background for the picker
            panel_padding = padding // 2
            overlay_bg_x1 = picker_x1 - panel_padding
            overlay_bg_y1 = picker_y1 - panel_padding
            overlay_bg_x2 = picker_x1 + picker_width + panel_padding
            overlay_bg_y2 = picker_y1 + picker_height + padding * 2 # More space for instruction below picker

            self._draw_transparent_rectangle(image, (overlay_bg_x1, overlay_bg_y1, overlay_bg_x2, overlay_bg_y2),
                                             self.UI_PANEL_BG_COLOR, self.UI_PANEL_ALPHA)

            # Draw the color spectrum image, scaled to current picker dimensions
            scaled_spectrum = cv2.resize(self.color_spectrum_image, (picker_width, picker_height), interpolation=cv2.INTER_LINEAR)
            image[picker_y1:picker_y1+picker_height, picker_x1:picker_x1+picker_width] = scaled_spectrum
            cv2.rectangle(image, (picker_x1, picker_y1), (picker_x1+picker_width, picker_y1+picker_height), self.UI_TEXT_COLOR, 2)

            # Add instructions text within the overlay area
            instruction_text_picker = "Pick your color of your day"
            
            text_size = cv2.getTextSize(instruction_text_picker, cv2.FONT_HERSHEY_SIMPLEX, subheader_fs, self.THICKNESS_NORMAL)[0]
            text_x = picker_x1 + (picker_width - text_size[0]) // 2
            text_y = picker_y1 + picker_height + text_size[1] + panel_padding # Position below the spectrum image

            cv2.putText(image, instruction_text_picker, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, subheader_fs, self.UI_TEXT_COLOR, self.THICKNESS_NORMAL, cv2.LINE_AA)

            # Draw current selected color point on the palette
            indicator_x_abs = picker_x1 + int(self.last_picked_relative_pos_on_picker[0] * picker_width)
            indicator_y_abs = picker_y1 + int(self.last_picked_relative_pos_on_picker[1] * picker_height)

            cv2.circle(image, (indicator_x_abs, indicator_y_abs), int(padding * 0.4), self.UI_TEXT_COLOR, 2) # Outer border
            cv2.circle(image, (indicator_x_abs, indicator_y_abs), int(padding * 0.3), self.current_overlay_color, -1) # Inner color


        # --- General Instructions (Bottom Left) ---
        instructions_text = [
            "Instructions:",
            f"  - Peace Sign (or 'c') to toggle Color Picker ({self.current_gesture_cooldown} frames)",
            "  - Pinch to select a color",
            f"  - Release pinch (after picking) to generate poem ({self.poem_generation_cooldown} frames)",
            "  - Press 'q' to Quit"
        ]
        
        # Calculate total height of instructions block for positioning
        total_instructions_height = 0
        for i, line in enumerate(instructions_text):
            font_scale = subheader_fs if i == 0 else body_fs
            thickness = self.THICKNESS_BOLD if i == 0 else self.THICKNESS_NORMAL
            total_instructions_height += int(cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][1] * self.LINE_HEIGHT_FACTOR)
        
        current_y_inst = img_height - padding - total_instructions_height
        
        for i, line in enumerate(instructions_text):
            color = self.UI_TEXT_COLOR
            font_scale = body_fs
            thickness = self.THICKNESS_NORMAL
            if i == 0:
                color = self.UI_ACCENT_COLOR # Highlight title
                font_scale = subheader_fs
                thickness = self.THICKNESS_BOLD

            cv2.putText(image, line, (padding, current_y_inst),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            current_y_inst += int(cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][1] * self.LINE_HEIGHT_FACTOR)
        
        # Draw Cooldown Bars near their respective instructions
        cooldown_bar_height = int(padding * 0.5)
        cooldown_bar_width = int(img_width * 0.15) # Example width for bar

        # Gesture Cooldown Bar
        # Get exact Y position of the "Peace Sign" line
        peace_sign_line_y_offset = int(cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, subheader_fs, self.THICKNESS_BOLD)[0][1] * self.LINE_HEIGHT_FACTOR) # Height of "Instructions:" line
        peace_sign_line_y = (img_height - padding - total_instructions_height) + peace_sign_line_y_offset
        
        bar_x = padding + cv2.getTextSize(instructions_text[1].split('(')[0], cv2.FONT_HERSHEY_SIMPLEX, body_fs, self.THICKNESS_NORMAL)[0][0] + padding // 2
        bar_y = peace_sign_line_y - int(padding * 0.2) # Adjust slightly up relative to its line

        # self._draw_cooldown_bar(image, bar_x, bar_y, cooldown_bar_width, cooldown_bar_height,
        #                         self.current_gesture_cooldown, self.GESTURE_TOGGLE_COOLDOWN_FRAMES,
        #                         color=self.UI_ACCENT_COLOR if self.current_gesture_cooldown > 0 else (50,50,50))

        # Poem Generation Cooldown Bar
        # Get exact Y position of the "Release pinch" line
        poem_gen_line_y_offset = peace_sign_line_y_offset + \
                                 int(cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, body_fs, self.THICKNESS_NORMAL)[0][1] * self.LINE_HEIGHT_FACTOR) + \
                                 int(cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, body_fs, self.THICKNESS_NORMAL)[0][1] * self.LINE_HEIGHT_FACTOR) # Height of "Instructions:", "Peace Sign", "Pinch to select"
        poem_gen_line_y = (img_height - padding - total_instructions_height) + poem_gen_line_y_offset

        bar_x = padding + cv2.getTextSize(instructions_text[3].split('(')[0], cv2.FONT_HERSHEY_SIMPLEX, body_fs, self.THICKNESS_NORMAL)[0][0] + padding // 2
        bar_y = poem_gen_line_y - int(padding * 0.2) # Adjust slightly up relative to its line

        # self._draw_cooldown_bar(image, bar_x, bar_y, cooldown_bar_width, cooldown_bar_height,
        #                         self.poem_generation_cooldown, self.POEM_GENERATION_COOLDOWN_FRAMES,
        #                         color=self.UI_ACCENT_COLOR if self.poem_generation_cooldown > 0 else (50,50,50))


    def run(self):
        """Main application loop to capture frames and process gestures."""
        self.initialize_app() # Call the new initialization method

        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            self.total_frames += 1
            # Adjust timestamp for smoother flow; 33ms is approx 30 FPS
            self.frame_timestamp_ms = int(time.time() * 1000) 
            image = cv2.flip(image, 1)

            # Store previous frame's picking state for deactivation logic
            self.was_hand_picking_color_in_prev_frame = self.is_hand_picking_color
            self.is_peace_sign_detected_this_frame = False # Reset for current frame
            self.any_hand_detected_this_frame = False # Reset hand presence for current frame

            # Convert image to MediaPipe format and recognize gestures
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            recognition_result = self.recognizer.recognize_for_video(mp_image, self.frame_timestamp_ms)

            # Process hand landmarks and gestures
            if recognition_result.hand_landmarks:
                self._process_hand_landmarks(image, recognition_result)
            else:
                self.gesture_counts["None"] += 1 # No hands detected
                self.any_hand_detected_this_frame = False # No hands means not detected
                
            # Auto-deactivation logic for color picker AND NEW poem generation trigger
            # This triggers when the hand stops picking a color after actively doing so,
            # which signifies the completion of a color selection.
            if self.is_color_picker_active and self.was_hand_picking_color_in_prev_frame and not self.is_hand_picking_color:
                # User just released pinch after picking a color. Trigger poem generation.
                # Only generate if color changed and not already generating, and cooldown is ready
                if (self.current_overlay_color != self.last_color_for_poem_gen) and \
                   (not self.is_generating_poem) and \
                   (self.poem_generation_cooldown <= 0):
                    
                    self._start_poem_generation_thread(self.mood_text, self.warmth_text)
                    self.last_color_for_poem_gen = self.current_overlay_color
                    self.poem_generation_cooldown = self.POEM_GENERATION_COOLDOWN_FRAMES # Reset cooldown
                
                # Now, deactivate the color picker as the interaction is complete
                self.is_color_picker_active = False

            # Handle toggle gesture (Peace Sign) cooldown
            if self.current_gesture_cooldown > 0:
                self.current_gesture_cooldown -= 1

            # Handle poem generation cooldown (decrements even during generation)
            if self.poem_generation_cooldown > 0:
                self.poem_generation_cooldown -= 1

            # Toggle color picker with Peace Sign gesture
            if self.is_peace_sign_detected_this_frame and not self.gesture_active_in_prev_frame and self.current_gesture_cooldown <= 0:
                self.is_color_picker_active = not self.is_color_picker_active
                self.current_gesture_cooldown = self.GESTURE_TOGGLE_COOLDOWN_FRAMES
                self.is_hand_picking_color = False
                self.was_hand_picking_color_in_prev_frame = False
                # If color picker is deactivated, force poem regeneration next time it's active and color changes
                self.last_color_for_poem_gen = (-1, -1, -1) # dummy value to ensure color is "different"
                # If toggling off, stop any pending poem generation and clear message
                if not self.is_color_picker_active and self.is_generating_poem:
                    # In a more robust system, you'd send a signal to the thread to stop
                    # For simplicity, here we just acknowledge it will finish and update later.
                    self.is_generating_poem = False
                    self.generated_poem = "Poem generation cancelled."
                    if self.poem_thread and self.poem_thread.is_alive():
                        # A better way would be to make the thread stoppable, but for now, let it finish or join.
                        # For now, we'll let it finish in the background if it's already started.
                        pass


            self.gesture_active_in_prev_frame = self.is_peace_sign_detected_this_frame

            # Draw all UI elements on the image
            self._draw_ui_elements(image)

            # Show the final image
            cv2.imshow('Tarot', image) # Reverted window title

            # Handle Key Presses
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.is_color_picker_active = not self.is_color_picker_active
                self.is_hand_picking_color = False
                self.was_hand_picking_color_in_prev_frame = False
                self.current_gesture_cooldown = self.GESTURE_TOGGLE_COOLDOWN_FRAMES
                # If color picker is toggled via 'c', force poem regeneration next time
                self.last_color_for_poem_gen = (-1, -1, -1) # dummy value to ensure color is "different"
                # If toggling off, stop any pending poem generation and clear message
                if not self.is_color_picker_active and self.is_generating_poem:
                    self.is_generating_poem = False
                    self.generated_poem = "Poem generation cancelled."
                    if self.poem_thread and self.poem_thread.is_alive():
                        pass # Let it finish, or implement a stop signal if needed.

        # Ensure the camera is released and windows are destroyed even if a thread is running
        self.cap.release()
        cv2.destroyAllWindows()
        # Optional: wait for the poem thread to finish before truly exiting
        if self.poem_thread and self.poem_thread.is_alive():
            print("Waiting for poem generation thread to finish...")
            self.poem_thread.join(timeout=5) # Wait for max 5 seconds
            if self.poem_thread.is_alive():
                print("Poem thread did not finish gracefully.")

# --- Main execution block ---
if __name__ == "__main__":
    app = HandGestureControlApp()
    app.run()