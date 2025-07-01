import os
import math
import uuid
from PIL import Image

def create_collage_from_folder(folder_name='frames', output_subfolder='collages',
                                image_size=(320, 240), padding=10, bg_color=(255, 255, 255)):
    """
    Creates a collage from all image files in a folder and saves it in the same base directory as the script.

    Parameters:
    - folder_name: Relative path to folder containing input images (e.g., "accident_frames")
    - output_subfolder: Subfolder name where collage will be saved (e.g., "collages")
    - image_size: Size of each image in the collage (width, height)
    - padding: Space between images
    - bg_color: RGB background color for the collage

    Returns:
    - Path to the saved collage image
    """
    # Get absolute path to the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Absolute paths to input and output folders
    input_folder = os.path.join(base_dir, folder_name)
    output_folder = os.path.join(base_dir, output_subfolder)

    # List and validate images
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [os.path.join(input_folder, f) for f in sorted(os.listdir(input_folder)) if f.lower().endswith(valid_exts)]
    
    num_images = len(image_files)
    if num_images == 0:
        raise ValueError("No valid image files found in the folder.")

    # Auto calculate rows and cols based on number of images
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    collage_width = cols * image_size[0] + (cols + 1) * padding
    collage_height = rows * image_size[1] + (rows + 1) * padding
    collage_img = Image.new('RGB', (collage_width, collage_height), bg_color)

    # Paste each image into the collage
    for i, img_path in enumerate(image_files):
        img = Image.open(img_path).resize(image_size)
        x = padding + (i % cols) * (image_size[0] + padding)
        y = padding + (i // cols) * (image_size[1] + padding)
        collage_img.paste(img, (x, y))

    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Generate unique filename using UUID
    unique_id = str(uuid.uuid4())[:8]
    collage_filename = f'accident_snapshot_collage_{unique_id}.jpg'
    collage_path = os.path.join(output_folder, collage_filename)
    
    collage_img.save(collage_path)
    print(f"✅ Collage saved at: {collage_path}")

    return collage_path

# Example usage
if __name__ == "__main__":
    create_collage_from_folder()
