import base64
import sys
import os
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_runner.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # Create the JSON input structure
    test_input = {
        "input": {
            "image": encoded_string
        }
    }

    # Construct the command for local testing
    # We wrap the JSON string in single quotes to handle shell interpretation
    test_command = f"python handler.py --test_input '{json.dumps(test_input)}'"

    print("Execute the following command in your pod's terminal to test:")
    print("-" * 70)
    print(test_command)
    print("-" * 70)

if __name__ == "__main__":
    main()