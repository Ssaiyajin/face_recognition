import face_recognition
import os
import cv2
import pickle
import time

Known_Faces_Dir = 'known_faces_1'
Tolerance = 0.7
Frame_Thickness = 3
Font_Thickness = 2
Model = 'hog'
video = cv2.VideoCapture(0)

def name_to_color(name):
    """Generate a color based on the first three letters of the name."""
    return [(ord(c.lower()) - 97) * 8 for c in name[:3]]

def load_known_faces(directory):
    """Load known faces and their names from the specified directory."""
    known_faces = []
    knows_names = []
    
    for name in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, name)):
            with open(os.path.join(directory, name, filename), "rb") as f:
                encoding = pickle.load(f)
            known_faces.append(encoding)
            knows_names.append(int(name))

    return known_faces, knows_names

def main():
    print("Loading known faces...")
    known_faces, knows_names = load_known_faces(Known_Faces_Dir)

    if known_faces:
        next_id = max(map(int, knows_names)) + 1
    else:
        next_id = 1

    print("Processing video...")
    
    while True:
        ret, image = video.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Check the shape and type of the captured image
        print(f"Captured image shape: {image.shape} and dtype: {image.dtype}")

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Check the shape and type after conversion
        print(f"Converted image shape: {image_rgb.shape} and dtype: {image_rgb.dtype}")

        # Ensure image is 8-bit and RGB
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != 'uint8':
            print("Image is not in the correct format. Skipping this frame.")
            continue

        locations = face_recognition.face_locations(image_rgb, model=Model)
        encodings = face_recognition.face_encodings(image_rgb, locations)

        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, Tolerance)
            match = None
            
            if True in results:
                match = knows_names[results.index(True)]
                print(f"Match found: {match}")
            else:
                match = str(next_id)
                next_id += 1
                knows_names.append(match)
                known_faces.append(face_encoding)
                os.makedirs(os.path.join(Known_Faces_Dir, match), exist_ok=True)
                with open(os.path.join(Known_Faces_Dir, match, f"{match}-{int(time.time())}.pkl"), "wb") as f:
                    pickle.dump(face_encoding, f)

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = name_to_color(match)
            cv2.rectangle(image, top_left, bottom_right, color, Frame_Thickness)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), Font_Thickness)

        cv2.imshow("Face Recognition", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    video.release() 
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()