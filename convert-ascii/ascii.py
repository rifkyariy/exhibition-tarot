import os
import json
from datetime import datetime
from PIL import Image, ImageEnhance

# --- Configuration ---
# Directory where your source images are located.
IMAGE_DIRECTORY = "images"

# The name of the final JSON animation file to be created.
OUTPUT_JSON_FILE = "animation.json"

# --- Animation Metadata ---
# Desired playback speed in frames per second (FPS).
ANIMATION_FPS = 24
# The sequence range of your image files.
START_NUMBER = 1
END_NUMBER =360

# --- ASCII Conversion Settings ---
# A 70-character set for smooth gradients, from darkest to lightest.# Lightest to Darkest (Inverted)
ASCII_CHARS = list(" .'`^\",:;Il!i><~+_-?][}{)(|/\\tfjrxnuvczXYUJCQL0OZmwqpdbkho*#MW&8%B@$")
# Set the desired width of the output ASCII art in characters.
OUTPUT_WIDTH = 400
# Set the threshold for transparency (0=fully transparent, 255=fully opaque).
ALPHA_THRESHOLD = 142
# Adjusts the brightness. 1.0 is original, >1.0 is brighter.
BRIGHTNESS_FACTOR = 1
# Adjusts the contrast. 1.0 is original, >1.0 is higher contrast.
CONTRAST_FACTOR = 1.5
# --------------------

def image_to_ascii(image_path):
    """
    Converts a single image file to a high-contrast ASCII art string.
    
    Args:
        image_path (str): The full path to the input image.

    Returns:
        str: The generated ASCII art as a multi-line string, or None on error.
    """
    try:
        source_image = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"    -> ❌ Error opening image: {e}")
        return None

    # Resize image
    width, height = source_image.size
    aspect_ratio = height / float(width)
    new_height = int(aspect_ratio * OUTPUT_WIDTH * 0.55)
    resized_image = source_image.resize((OUTPUT_WIDTH, new_height))

    # Enhance a grayscale version for contrast and brightness
    grayscale_image = resized_image.convert("L")
    bright_image = ImageEnhance.Brightness(grayscale_image).enhance(BRIGHTNESS_FACTOR)
    final_image = ImageEnhance.Contrast(bright_image).enhance(CONTRAST_FACTOR)

    # Map pixels to ASCII characters
    rgba_pixels = resized_image.getdata()
    grayscale_pixels = final_image.getdata()
    
    ascii_chars_list = []
    for rgba_pixel, gray_pixel_value in zip(rgba_pixels, grayscale_pixels):
        if rgba_pixel[3] < ALPHA_THRESHOLD:
            ascii_chars_list.append(" ")
            continue

        char_index = int(gray_pixel_value / 256 * len(ASCII_CHARS))
        if char_index >= len(ASCII_CHARS):
            char_index = len(ASCII_CHARS) - 1
        ascii_chars_list.append(ASCII_CHARS[char_index])

    ascii_str = "".join(ascii_chars_list)

    # Format into lines
    img_ascii = ""
    for i in range(0, len(ascii_str), OUTPUT_WIDTH):
        img_ascii += ascii_str[i:i + OUTPUT_WIDTH] + "\n"

    return img_ascii

def create_animation_from_images():
    """
    Reads a sequence of images, converts them to ASCII art, and compiles
    them into a single JSON animation file.
    """
    print("🚀 Starting animation creation from images...")
    
    frames_data = []
    width, height = 0, 0

    # --- Loop through the source image files ---
    for i in range(START_NUMBER, END_NUMBER + 1):
        filename = f"{i:04d}.png"
        filepath = os.path.join(IMAGE_DIRECTORY, filename)

        print(f"Processing {filename}...")

        if not os.path.exists(filepath):
            print(f"    -> ⚠️  Warning: Skipping missing file {filename}")
            continue

        # Convert image directly to an ASCII string
        frame_content = image_to_ascii(filepath)
        
        if not frame_content:
            print(f"    -> ❌ Error: Failed to convert {filename}")
            continue

        # If this is the first successful frame, determine dimensions from it.
        if not frames_data:
            lines = frame_content.splitlines()
            height = len(lines)
            if height > 0:
                width = len(lines[0])
        
        frames_data.append(frame_content)
    
    if not frames_data:
        print("❌ Error: No frames were processed. Check the image directory and file names.")
        return

    print(f"\n✅ Found and processed {len(frames_data)} frames.")
    print(f"Detected dimensions: {width}x{height} characters.")

    # --- Assemble the final JSON object ---
    animation_data = {
        "metadata": {
            "title": "My ASCII Art Animation",
            "width": width,
            "height": height,
            "frameCount": len(frames_data),
            "fps": ANIMATION_FPS,
            "createdAt": datetime.utcnow().isoformat() + "Z"
        },
        "frames": frames_data
    }

    # --- Write the JSON file ---
    print(f"💾 Saving animation to {OUTPUT_JSON_FILE}...")
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(animation_data, f, indent=2)

    print("\n✨ JSON animation created successfully!")

if __name__ == "__main__":
    # Ensure the Pillow library is installed: pip install Pillow
    create_animation_from_images()
