from threading import Thread
import cv2
import time
from queue import Queue


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


class FileVideoStream:
    def __init__(self, path, transform=None, queue_size=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.transform = transform

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stopped = True

                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform:
                    frame = self.transform(frame)

                # add the frame to the queue
                self.Q.put(frame)
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()
