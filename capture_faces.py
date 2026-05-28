import cv2
import os
import time

def capture_faces():
    train_dir = "dataset/train/thankyou"
    val_dir = "dataset/val/thankyou"
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Load OpenCV's built-in face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(1) # FaceTime HD Camera
    if not cap.isOpened():
        print("[Error] Could not open camera 1")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("=========================================")
    print("  Face Data Collection Script")
    print("=========================================")
    print("1. Look at the camera.")
    print("2. The script will automatically detect your face.")
    print("3. It will capture 100 images for training, and 20 for validation.")
    print("4. Move your head around slightly during capture for better data.")
    print("Press 'C' to start capturing, or 'Q' to quit.")
    
    captured_train = 0
    captured_val = 0
    total_train = 100
    total_val = 20
    is_capturing = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        display_frame = frame.copy()
        
        # Draw face boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if is_capturing:
                # Add some margin to the face
                margin = int(w * 0.2)
                y1 = max(0, y - margin)
                y2 = min(frame.shape[0], y + h + margin)
                x1 = max(0, x - margin)
                x2 = min(frame.shape[1], x + w + margin)
                
                face_img = frame[y1:y2, x1:x2]
                
                # Resize to model format
                if face_img.size > 0:
                    face_img = cv2.resize(face_img, (64, 64))
                    timestamp = time.time()
                    
                    if captured_train < total_train:
                        filepath = os.path.join(train_dir, f"face_{timestamp}.jpg")
                        cv2.imwrite(filepath, face_img)
                        captured_train += 1
                    elif captured_val < total_val:
                        filepath = os.path.join(val_dir, f"face_{timestamp}.jpg")
                        cv2.imwrite(filepath, face_img)
                        captured_val += 1
                    else:
                        print("\n[Success] Capture complete!")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                        
        # Overlay UI
        if not is_capturing:
            cv2.putText(display_frame, "Press 'C' to start capturing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            total = total_train + total_val
            current = captured_train + captured_val
            progress = current / total
            cv2.rectangle(display_frame, (10, 20), (int(10 + 600 * progress), 40), (0, 255, 0), -1)
            cv2.putText(display_frame, f"Captured: {current}/{total}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        cv2.imshow("Face Capture", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not is_capturing:
            is_capturing = True
            print("Capturing started! Move your head slightly...")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_faces()
