import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import numpy as np

# Import the landmark_pb2 module for drawing utilities
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
    COLOR_PICKER_WIDTH = 600
    COLOR_PICKER_HEIGHT = 350
    GESTURE_TOGGLE_COOLDOWN_FRAMES = 20 # Cooldown for peace sign toggle

    def __init__(self):
        """Initializes the MediaPipe recognizer, OpenCV camera, and application state variables."""
        # MediaPipe setup
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        # OpenCV setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()
        cv2.namedWindow('Hand Gesture Control', cv2.WINDOW_AUTOSIZE)

        # MediaPipe Drawing Utilities
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
        self.last_picked_relative_pos_on_picker = (self.COLOR_PICKER_WIDTH // 2, self.COLOR_PICKER_HEIGHT // 2)

        self.current_gesture_cooldown = 0
        self.gesture_active_in_prev_frame = False

        self.total_frames = 0
        self.gesture_counts = {
            "Thumb Up": 0, "Thumb Down": 0, "Peace Sign": 0,
            "Index Up": 0, "Closed": 0, "Open": 0, "I Love You": 0,
            "Pinching": 0, "None": 0
        }
        self.frame_timestamp_ms = 0

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
        start_hue_color_bgr = np.array([255, 0, 0], dtype=np.float32)  # Blue (cool)
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
                    picker_x1, picker_y1, picker_x2, picker_y2 = self._get_color_picker_coords(image)
                    if picker_x1 <= self.current_drag_pos[0] <= picker_x2 and \
                       picker_y1 <= self.current_drag_pos[1] <= picker_y2:
                        self.is_hand_picking_color = True
                        relative_x = self.current_drag_pos[0] - picker_x1
                        relative_y = self.current_drag_pos[1] - picker_y1
                        relative_x = np.clip(relative_x, 0, self.COLOR_PICKER_WIDTH - 1)
                        relative_y = np.clip(relative_y, 0, self.COLOR_PICKER_HEIGHT - 1)

                        if self.color_spectrum_image is not None:
                            self.current_overlay_color = tuple(self.color_spectrum_image[relative_y, relative_x].tolist())
                            self.current_color_name = "Custom"
                            self.last_picked_relative_pos_on_picker = (relative_x, relative_y)

                if self.drag_start_pos is None:
                    self.drag_start_pos = self.current_drag_pos
            else:
                pass # This hand is not pinching

            # Display Handedness and Detected Gesture
            handedness_label = recognition_result.handedness[hand_index][0].category_name
            index_finger_mcp_landmark = hand_landmarks_list[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
            text_x = int(index_finger_mcp_landmark.x * image.shape[1])
            text_y = int(index_finger_mcp_landmark.y * image.shape[0]) - 30 - (hand_index * 40)
            cv2.putText(image, f"{handedness_label} Hand: {current_hand_gesture_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
        """Calculates and returns the coordinates (x1, y1, x2, y2) of the color picker."""
        img_height, img_width, _ = image.shape
        picker_x1 = (img_width - self.COLOR_PICKER_WIDTH) // 2
        picker_y1 = img_height - self.COLOR_PICKER_HEIGHT - 10 # Padding from bottom
        picker_x2 = picker_x1 + self.COLOR_PICKER_WIDTH
        picker_y2 = picker_y1 + self.COLOR_PICKER_HEIGHT
        return picker_x1, picker_y1, picker_x2, picker_y2

    def _draw_ui_elements(self, image):
        """Draws various UI elements on the image."""
        # Visual feedback for drag:
        if self.drag_start_pos and self.current_drag_pos and self.any_hand_pinching_this_frame:
            cv2.line(image, self.drag_start_pos, self.current_drag_pos, self.current_overlay_color, 3)
            cv2.putText(image, "Dragging", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, self.current_overlay_color, 2)

        # Display Current Overlay Color UI and instructions
        cv2.putText(image, f"Overlay Color: {self.current_color_name}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Press 'c' or use Peace Sign gesture to toggle color picker", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display Gesture Percentages
        y_offset_start = 200
        line_height = 30
        for i, (gesture_name, count) in enumerate(self.gesture_counts.items()):
            percentage = (count / self.total_frames) * 100 if self.total_frames > 0 else 0
            cv2.putText(image, f"{gesture_name}: {percentage:.1f}%",
                        (10, y_offset_start + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw Color Picker Overlay if active
        if self.is_color_picker_active:
            picker_x1, picker_y1, picker_x2, picker_y2 = self._get_color_picker_coords(image)

            # Generate color spectrum image only once or if dimensions change
            if self.color_spectrum_image is None or \
               self.color_spectrum_image.shape[1] != self.COLOR_PICKER_WIDTH or \
               self.color_spectrum_image.shape[0] != self.COLOR_PICKER_HEIGHT:
                self.color_spectrum_image = self._generate_custom_gradient_image(self.COLOR_PICKER_WIDTH, self.COLOR_PICKER_HEIGHT)

            # Draw semi-transparent dark background for the picker
            overlay_bg_x1 = picker_x1 - 10
            overlay_bg_y1 = picker_y1 - 10
            overlay_bg_x2 = picker_x2 + 10
            overlay_bg_y2 = picker_y2 + 40 # Added space for text below

            overlay_bg_x1 = max(0, overlay_bg_x1)
            overlay_bg_y1 = max(0, overlay_bg_y1)
            overlay_bg_x2 = min(image.shape[1], overlay_bg_x2)
            overlay_bg_y2 = min(image.shape[0], overlay_bg_y2)

            overlay_roi = image[overlay_bg_y1 : overlay_bg_y2, overlay_bg_x1 : overlay_bg_x2].copy()
            cv2.rectangle(overlay_roi, (0, 0), (overlay_roi.shape[1], overlay_roi.shape[0]), (50, 50, 50), -1)
            alpha = 0.6
            image[overlay_bg_y1 : overlay_bg_y2, overlay_bg_x1 : overlay_bg_x2] = cv2.addWeighted(
                overlay_roi, alpha, image[overlay_bg_y1 : overlay_bg_y2, overlay_bg_x1 : overlay_bg_x2], 1 - alpha, 0
            )

            # Draw the color spectrum image
            image[picker_y1:picker_y2, picker_x1:picker_x2] = self.color_spectrum_image
            cv2.rectangle(image, (picker_x1, picker_y1), (picker_x2, picker_y2), (200, 200, 200), 2)

            # Add instructions text within the overlay area
            instruction_text = "Pinch and Drag hand to pick a color of your day"
            text_font_scale = 0.7
            text_thickness = 2
            text_color = (255, 255, 255)

            text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, text_thickness)[0]
            text_x = picker_x1 + (self.COLOR_PICKER_WIDTH - text_size[0]) // 2
            text_y = picker_y2 + 40 # Position below the spectrum image with increased padding

            cv2.putText(image, instruction_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, text_color, text_thickness)

            # Draw current selected color point on the palette
            if self.is_hand_picking_color and self.current_drag_pos:
                indicator_x_abs = self.current_drag_pos[0]
                indicator_y_abs = self.current_drag_pos[1]
            else:
                indicator_x_abs = picker_x1 + self.last_picked_relative_pos_on_picker[0]
                indicator_y_abs = picker_y1 + self.last_picked_relative_pos_on_picker[1]

            cv2.circle(image, (indicator_x_abs, indicator_y_abs), 10, (255, 255, 255), 2)
            cv2.circle(image, (indicator_x_abs, indicator_y_abs), 8, self.current_overlay_color, -1)

    def run(self):
        """Main application loop to capture frames and process gestures."""
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            self.total_frames += 1
            self.frame_timestamp_ms += 33
            image = cv2.flip(image, 1)

            # Store previous frame's picking state for deactivation logic
            self.was_hand_picking_color_in_prev_frame = self.is_hand_picking_color
            self.is_peace_sign_detected_this_frame = False # Reset for current frame

            # Convert image to MediaPipe format and recognize gestures
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            recognition_result = self.recognizer.recognize_for_video(mp_image, self.frame_timestamp_ms)

            # Process hand landmarks and gestures
            if recognition_result.hand_landmarks:
                self._process_hand_landmarks(image, recognition_result)
            else:
                self.gesture_counts["None"] += 1 # No hands detected

            # Auto-deactivation logic for color picker
            if self.is_color_picker_active and self.was_hand_picking_color_in_prev_frame and not self.is_hand_picking_color:
                self.is_color_picker_active = False

            # Handle toggle gesture (Peace Sign) cooldown
            if self.current_gesture_cooldown > 0:
                self.current_gesture_cooldown -= 1

            # Toggle color picker with Peace Sign gesture
            if self.is_peace_sign_detected_this_frame and not self.gesture_active_in_prev_frame and self.current_gesture_cooldown <= 0:
                self.is_color_picker_active = not self.is_color_picker_active
                self.current_gesture_cooldown = self.GESTURE_TOGGLE_COOLDOWN_FRAMES
                self.is_hand_picking_color = False
                self.was_hand_picking_color_in_prev_frame = False

            self.gesture_active_in_prev_frame = self.is_peace_sign_detected_this_frame

            # Draw all UI elements on the image
            self._draw_ui_elements(image)

            # Show the final image
            cv2.imshow('Tarot', image) # Changed window title to 'Tarot' as per implicit request

            # Handle Key Presses
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.is_color_picker_active = not self.is_color_picker_active
                self.is_hand_picking_color = False
                self.was_hand_picking_color_in_prev_frame = False
                self.current_gesture_cooldown = self.GESTURE_TOGGLE_COOLDOWN_FRAMES

        self.cap.release()
        cv2.destroyAllWindows()

# --- Main execution block ---
if __name__ == "__main__":
    app = HandGestureControlApp()
    app.run()
