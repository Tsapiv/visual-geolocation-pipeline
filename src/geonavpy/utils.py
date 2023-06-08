import base64
import hashlib
import hmac
import os
import re
import urllib.parse as urlparse
from typing import List

import cv2
import numpy as np
from matplotlib import cm


def sign_url(input_url=None, secret=None):
    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")

    url = urlparse.urlparse(input_url)
    url_to_sign = url.path + "?" + url.query
    decoded_key = base64.urlsafe_b64decode(secret)
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)
    encoded_signature = base64.urlsafe_b64encode(signature.digest())
    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query
    return original_url + "&signature=" + encoded_signature.decode()


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_ordering(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def match_paths(rootdir: str, pattern: str):
    regex = re.compile(pattern)
    paths = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if regex.match(file):
                paths.append(os.path.join(root, file))
    return sorted(paths, key=natural_ordering)


def make_retrieval_plot(query_image: np.ndarray, retrieved_images: List[np.ndarray]):
    shapes = np.asarray(list(map(lambda x: x.shape[:2], retrieved_images + [query_image])))
    max_h, max_w = np.max(shapes, axis=0)
    anchor = query_image
    anchor_padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    anchor_padded[:anchor.shape[0], :anchor.shape[1], :] = anchor
    window = f'Retrieval'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    for rank, image in enumerate(retrieved_images):
        image_padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        image_padded[:image.shape[0], :image.shape[1], :] = image
        cv2.imshow(window, np.concatenate((anchor_padded, image_padded), axis=1))
        cv2.waitKey()
    cv2.destroyAllWindows()

def make_keypoints_plot(image, kpts):
    out = np.copy(image)
    kpts = np.round(kpts).astype(int)
    white = (255, 255, 255)
    black = (0, 0, 0)
    for x, y in kpts:
        cv2.circle(out, (x, y), 4, black, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), 2, white, -1, lineType=cv2.LINE_AA)
    return out


def make_matching_plot(image0, image1, kpts0, kpts1, matches, match_confidences, text=(), path=None,
                       show_keypoints=False, margin=10,
                       opencv_display=False, opencv_title='',
                       small_text=()):
    H0, W0 = image0.shape[:2]
    H1, W1 = image1.shape[:2]
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0 + margin:, :] = image1

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = cm.jet(match_confidences[valid])

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(0)

    return out
