import sys
import numpy as np
import cv2

FLANN_INDEX_KDTREE = 1 # bug: flann enums are missing

def init_feature():
    detector = cv2.SIFT(1000)
    norm = cv2.NORM_L2
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
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
        cv2.polylines(vis, [corners], True, (255, 255, 255))
        # cv2.fillPoly(vis, [corners], (255, 255, 255))


    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs])

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x, y), inlier in zip(p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x, y), 2, col, -1)

    cv2.imshow(win, vis)

def main():
    arg1 = sys.argv[1]
    img1 = cv2.imread(arg1, 0)

    detector, matcher = init_feature()

    img1g = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    kp1, desc1 = detector.detectAndCompute(img1, None)

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

    count = 0
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # count += 1
        # if count % 30 == 0:
        #     kp2, desc2 = detector.detectAndCompute(frame, None)
        #     match_and_draw('find_obj', img1, frame, kp1, kp2, desc1, desc2)
        # else:
        #     cv2.imshow('find_obj', frame)

        kp2, desc2 = detector.detectAndCompute(frame, None)
        match_and_draw('find_obj', img1, frame, kp1, kp2, desc1, desc2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
