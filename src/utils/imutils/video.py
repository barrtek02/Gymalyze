import time
from threading import Thread
import cv2


class FPS:
    def __init__(self):
        # Store the start time, end time, and total number of frames
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # Start the timer
        self._start = time.time()
        return self

    def stop(self):
        # Stop the timer
        self._end = time.time()

    def update(self):
        # Increment the total number of frames examined during the start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # Ensure elapsed time calculation doesn't fail if _end is None
        if self._end is None:
            return time.time() - self._start
        return self._end - self._start

    def fps(self):
        # Prevent division by zero by checking if elapsed time is greater than 0
        if self.elapsed() > 0:
            return self._numFrames / self.elapsed()
        return 0  # Return 0 FPS if not enough time has passed


class WebcamVideoStream:
    def __init__(self, src=0):
        # Initialize the video camera stream and read the first frame from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # Variable used to indicate if the thread should be stopped
        self.stopped = False

    def start(self):
        # Start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while True:
            if self.stopped:
                return
            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the frame most recently read
        return self.frame

    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()
