import cv2

print("Testing cameras...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"Index {i}: Failed to open")
        continue
    
    # Try reading a frame
    ret, frame = cap.read()
    if ret:
        print(f"Index {i}: SUCCESS! Read frame of shape {frame.shape}")
    else:
        print(f"Index {i}: Opened, but failed to read frame")
    
    cap.release()
print("Done.")
