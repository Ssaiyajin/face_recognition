import face_recognition
import os
import cv2

KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

# Returns (R, G, B) from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    color = [(ord(c.lower()) - 97) * 8 for c in name[:3]]
    return color

print('Loading known faces...')
known_faces = []
known_names = []

# Check if the known faces directory exists and process each subdirectory
if os.path.isdir(KNOWN_FACES_DIR):
    for name in os.listdir(KNOWN_FACES_DIR):
        subfolder_path = os.path.join(KNOWN_FACES_DIR, name)
        
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            for filename in os.listdir(subfolder_path):
                # Construct full file path
                file_path = os.path.join(subfolder_path, filename)
                
                # Load an image
                image = face_recognition.load_image_file(file_path)

                # Get 128-dimension face encoding
                encodings = face_recognition.face_encodings(image)
                
                if encodings:  # Check if any face encodings were found
                    encoding = encodings[0]
                    known_faces.append(encoding)
                    known_names.append(name)

print('Processing unknown faces...')
if os.path.isdir(UNKNOWN_FACES_DIR):
    for filename in os.listdir(UNKNOWN_FACES_DIR):
        # Load image
        print(f'Filename {filename}', end='')
        image = face_recognition.load_image_file(os.path.join(UNKNOWN_FACES_DIR, filename))

        # Find face locations and encodings
        locations = face_recognition.face_locations(image, model=MODEL)
        encodings = face_recognition.face_encodings(image, locations)

        # Convert the image from RGB (face_recognition) to BGR (OpenCV)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        print(f', found {len(encodings)} face(s)')
        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

            match = None
            if True in results:  # If at least one is true, get the name of the first found label
                match = known_names[results.index(True)]
                print(f' - {match} from {results}')

                # Draw rectangle around the face
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])
                color = name_to_color(match)

                # Debug: Check the coordinates
                print(f"Drawing rectangle at: {top_left}, {bottom_right} with color: {color}")

                # Draw rectangle around the face
                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

                # Draw a filled rectangle for the name background
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)

                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

        # Resize the image for display
        image_resized = cv2.resize(image, (800, 600))  # Adjust this size as needed
        cv2.imshow(filename, image_resized)
        cv2.waitKey(0)
        cv2.destroyWindow(filename)
else:
    print(f"Directory {UNKNOWN_FACES_DIR} does not exist.")
