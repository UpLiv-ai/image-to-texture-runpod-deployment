import base64
import sys
import os
import json
from PIL import Image
from io import BytesIO

# Import the handler function from your handler.py file
import handler

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_runner.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)

    print(f"Processing image: {image_path}")

    # 1. Prepare the job input
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    job = {
        "input": {
            "image": encoded_string
        }
    }

    # 2. Run the handler function directly
    result = handler.handler(job)

    # 3. Process the result
    if "error" in result:
        print(f"An error occurred: {result['error']}")
        return

    print("Job completed successfully. Saving output images...")

    # Create a directory to store the results for this image
    base_filename = os.path.splitext(os.path.basename(image_path))
    output_folder = f"output_{base_filename}"
    os.makedirs(output_folder, exist_ok=True)

    # 4. Decode and save each output image
    for key, b64_string in result.items():
        if key.endswith("_b64"):
            try:
                image_data = base64.b64decode(b64_string)
                image = Image.open(BytesIO(image_data))
                
                # Construct a descriptive filename
                output_filename = f"{key.replace('_b64', '')}.png"
                output_path = os.path.join(output_folder, output_filename)
                
                image.save(output_path)
                print(f"  - Saved {output_path}")
            except Exception as e:
                print(f"Could not decode or save image for key '{key}': {e}")

    print(f"\nAll output images saved in the '{output_folder}/' directory.")

if __name__ == "__main__":
    main()