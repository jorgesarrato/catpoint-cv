import cv2
import os
import threading
from dotenv import load_dotenv

load_dotenv()

class TapoStream:
    def __init__(self):
        # Load from .env
        user = os.getenv("TAPO_USERNAME")
        password = os.getenv("TAPO_PASSWORD")
        ip = os.getenv("TAPO_IP")
        
        self.url = f"rtsp://{user}:{password}@{ip}:554/stream1"
        self.cap = cv2.VideoCapture(self.url)
        self.ret, self.frame = False, None
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
            else:
                self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


def main():
    tapo = TapoStream().start()
    
    print("Stream started. Press 'q' to quit.")

    while True:
        frame = tapo.read()

        if frame is not None:
            edges = cv2.Canny(frame, 100, 200)
            cv2.imshow("Threaded Tapo Feed", edges)

            #cv2.imshow("Threaded Tapo Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            tapo.stop()
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
