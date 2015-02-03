import sys
import numpy as np
import cv2
import time

class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

FLANN_INDEX_KDTREE = 1 # bug: flann enums are missing

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

def init_feature():
    detector = cv2.SIFT(1000)
    # norm = cv2.NORM_L2
    # flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3)
    # matcher = cv2.FlannBasedMatcher(flann_params, {})
    matcher = cv2.BFMatcher()
    return detector, matcher

def filter_matches(kp1, kp2, matches, ratio = 0.65):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = img2

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))
        # cv2.polylines(vis, [corners], True, (255, 255, 255))
        # cv2.fillPoly(vis, [corners], (255, 255, 255))

        mask = np.zeros(vis.shape, dtype=np.uint8)
        roi_corners = np.array([[(corners[0][0],corners[0][1]),
            (corners[1][0],corners[1][1]),
            (corners[2][0],corners[2][1]),
            (corners[3][0], corners[3][1])]], dtype=np.int32)
        white = (255, 255, 255)
        cv2.fillPoly(mask, roi_corners, white)

        # apply the mask
        masked_image = cv2.bitwise_and(vis, mask)

        # blurred_image = cv2.blur(vis, (15, 15), 0)
        blurred_image = cv2.boxFilter(vis, -1, (27, 27))
        vis = vis + (cv2.bitwise_and((blurred_image-vis), mask))

    # if status is None:
    #     status = np.ones(len(kp_pairs), np.bool_)
    # p2 = np.int32([kpp[1].pt for kpp in kp_pairs])

    # green = (0, 255, 0)
    # red = (0, 0, 255)
    # white = (255, 255, 255)
    # kp_color = (51, 103, 236)
    # for (x, y), inlier in zip(p2, status):
    #     if inlier:
    #         col = green
    #         cv2.circle(vis, (x, y), 2, col, -1)

    # view params
    width, height = 1280, 800
    x_offset = 260
    y_offset = 500
    l_img = create_blank(width, height, rgb_color=(0,0,0))

    vis = np.append(vis, vis, axis=1)
    vis = cv2.resize(vis, (0,0), fx=0.6, fy=0.6)

    l_img[y_offset:y_offset+vis.shape[0], x_offset:x_offset+vis.shape[1]] = vis

    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_AUTOSIZE, cv2.cv.CV_WINDOW_AUTOSIZE)
    cv2.imshow(win, l_img)

def main():
    imgs = [cv2.imread("images/ps1.jpg", 0),
            cv2.imread("images/linodeb.jpg", 0),
            #cv2.imread("images/google1.png", 0),
            #cv2.imread("images/hersheys1.png", 0),
            cv2.imread("images/luckycharm.jpg", 0),
            cv2.imread("images/starbucks1.jpg", 0),
            cv2.imread("images/mcdonalds.png", 0),
            cv2.imread("images/drpepper.png", 0),
            cv2.imread("images/spotify.jpg", 0),
            cv2.imread("images/pennapps.png", 0)]

    detector, matcher = init_feature()

    seeds = []
    for i in imgs:
        k, d = detector.detectAndCompute(i, None)
        seeds.append((k,d))

    def find_match(kp, desc):
        max_matches = 0
        match_img = imgs[0]
        match_desc= seeds[0]

        for i, seed in enumerate(seeds):
            raw_matches = matcher.knnMatch(desc, trainDescriptors = seed[1], k = 2)
            p1, p2, kp_pairs = filter_matches(kp, seed[0], raw_matches)
            if len(p1) > max_matches:
                max_matches = len(p1)
                match_desc = seed
                match_img = imgs[i]

        return (match_desc, match_img)


    def match_and_draw(win, img1, img2, kp1, kp2, desc1, desc2):
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2)
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        else:
            H, status = None, None

        vis = explore_match(win, img1, img2, kp_pairs, status, H)

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)


    while(cap.isOpened()):
        ret, frame = cap.read()

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # while True:
    #     ret, frame = cap.read()
    #     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #     # if count % 2 == 0 :
    #     #     kp2, desc2 = detector.detectAndCompute(frame, None)
    #     #     match_and_draw('find_obj', img1, frame, kp1, kp2, desc1, desc2)
    #     # else:
    #     #     cv2.imshow('find_obj', frame)

    #     # count += 1

    #     with Timer() as t:
    #         if count % 8 == 0:
    #             kp2, desc2 = detector.detectAndCompute(frame, None)
    #             (kp1, desc1), img1 = find_match(kp2, desc2)
    #     match_and_draw('Brand Killer', img1, frame, kp1, kp2, desc1, desc2)

    #     count += 1
    #     print('took %.03f sec.' % t.interval)


    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
