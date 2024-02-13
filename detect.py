import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    # Choose the model selection based on the detection requirement:
    #   - model_selection=0: Original model for near-range detection
    #   - model_selection=1: Model optimized for long-range detection
    face_detection = mp_face_detection.FaceDetection(
        min_detection_confidence=0.5,  # Minimum confidence threshold for detection
        model_selection=0)  # You can adjust parameters as needed

    # Initialize Video Capture
    cap = cv2.VideoCapture(0)  # You can adjust the camera index (0 for default webcam)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        results = face_detection.process(rgb_frame)

        # Draw bounding boxes around detected faces
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)
                # Add text on bounding box
                cv2.putText(frame, f'Face', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
