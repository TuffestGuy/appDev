import cv2
import os

fruit_classes = ['Orange', 'Bell Pepper', 'Pickle', 'Lemon', 'Carrot']
dataset_dir = 'dataset'

os.makedirs(dataset_dir, exist_ok=True)

for fruit in fruit_classes:
    class_dir = os.path.join(dataset_dir, fruit)
    os.makedirs(class_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print(f"Press 's' to capture images for {fruit}. Press 'q' to quit.")
    count = 0
    while True:
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            img_path = os.path.join(class_dir, f'fruit_{count}.jpg')
            cv2.imwrite(img_path, frame)
            print(f"Saved {img_path}")
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


