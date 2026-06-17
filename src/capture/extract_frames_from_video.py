import cv2
import os

def main():
    video_path = "data/raw_phone/video.mp4"
    output_dir = "data/raw_phone"

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Error opening video file")

    frame_count = 0
    save_every = 10  # adjust this (higher = fewer images)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % save_every == 0:
            filename = os.path.join(output_dir, f"frame_{frame_count:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

        frame_count += 1

    cap.release()
    print("Done extracting frames.")

if __name__ == "__main__":
    main()