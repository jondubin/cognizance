import cv2
import numpy as np

lk_params = dict(winSize=(15,15), maxLevel=2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))

subpix_params = dict(zeroZone=(-1,-1), winSize=(10,10),
                     criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,20,0.03))

feature_params = dict(maxCorners = 500, qualityLevel = 0.01, minDistance = 10)

class LKTracker(object):
    def __init__(self,imnames):
        self.imnames = imnames
        self.features = []
        self.tracks = []
        self.current_frame = 0

    def detect_points(self):
        self.image = cv2.imread(self.imnames[self.current_frame])
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        features = cv2.goodFeaturesToTrack(self.gray, **feature_params)

        cv2.cornerSubPix(self.gray, features, **subpix_params)

        self.features = features
        self.tracks = [[p] for p in features.reshape((-1,2))]

        self.prev_gray = self.gray

    def set_points(points, self):
        self.features = points
        self.tracks = [[p] for p in features.reshape((-1,2))]

    def track_points(self):
        if self.features != []:
            self.step()

            self.image = cv2.imread(self.imnames[self.current_frame])
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            tmp = np.float32(np.array(self.features).reshape(-1, 1, 2))

            features, status, track_error = cv2.calcOpticalFlowPyrLK(self.prev_gray,
                                                    self.gray, tmp, None, **lk_params)

            self.features = [p for (st,p) in zip(status, features) if st]

            features = features.reshape((-1,2))
            for i, f in enumerate(features):
                self.tracks[i].append(f)
            ndx = [i for (i, st) in enumerate(status) if not st]
            ndx.reverse()
            for i in ndx:
                self.tracks.pop(i)

            self.prev_gray = self.gray

    def step(self, framenbr = None):
        if framenbr is None:
            self.current_frame = (self.current_frame + 1) % len(self.imnames)
        else:
            self.current_frame = framenbr % len(self.imnames)

    def draw(self):
        for point in self.features:
            cv2.circle(self.image, (int(point[0][0]), int(point[0][1])), 3, (0,255,0), -1)

        cv2.imshow('LKtrack', self.image)
        cv2.waitKey()

imnames = ['images/corridor/bt.003.pgm', 'images/corridor/bt.002.pgm',
           'images/corridor/bt.001.pgm', 'images/corridor/bt.000.pgm']

lkt = LKTracker(imnames)

lkt.detect_points()
lkt.draw()
for i in range(len(imnames) - 1):
    lkt.track_points()
    lkt.draw()
