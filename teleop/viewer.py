"""
OpenCV window helpers: mouse orbit for the kitchen camera, text overlays, picture-in-picture.
"""
import cv2
import numpy as np


class OrbitControls:
    """
    Drag with the left mouse button in the kitchen window to orbit around the robot.
    Drag with the right mouse button to move up/down. Scroll to zoom.
    """

    def __init__(self, camera):
        self.camera = camera
        self.dragging = None
        self.last_xy = (0, 0)

    def on_mouse(self, event, x, y, flags, param):
        cam = self.camera
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self.dragging = "orbit" if event == cv2.EVENT_LBUTTONDOWN else "lift"
            self.last_xy = (x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.dragging = None
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx, dy = x - self.last_xy[0], y - self.last_xy[1]
            self.last_xy = (x, y)
            if self.dragging == "orbit":
                cam.azimuth -= 0.3 * dx
                cam.elevation = float(np.clip(cam.elevation - 0.3 * dy, -89, 89))
            else:
                cam.lookat[2] += 0.003 * dy
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom(0.9 if flags > 0 else 1.1)

    def zoom(self, factor):
        self.camera.distance = float(np.clip(self.camera.distance * factor, 0.3, 6.0))


def draw_text_lines(image, lines, corner="top-left", scale=0.55):
    """Write a few lines of text on a dark box so they are readable on any background."""
    if not lines:
        return image
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = int(28 * scale / 0.55)
    width = max(cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines) + 16
    height = line_height * len(lines) + 10

    h, w = image.shape[:2]
    x0 = 8 if "left" in corner else w - width - 8
    y0 = 8 if "top" in corner else h - height - 8

    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, dst=image)
    for i, line in enumerate(lines):
        y = y0 + 6 + line_height * (i + 1) - line_height // 4
        cv2.putText(image, line, (x0 + 8, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def paste_inset(image, small, width_fraction=0.28, corner="top-right"):
    """Paste a small picture into a corner of a bigger one (picture-in-picture)."""
    h, w = image.shape[:2]
    new_w = int(w * width_fraction)
    new_h = int(small.shape[0] * new_w / small.shape[1])
    small = cv2.resize(small, (new_w, new_h))
    x0 = w - new_w - 8 if "right" in corner else 8
    y0 = 8 if "top" in corner else h - new_h - 8
    image[y0:y0 + new_h, x0:x0 + new_w] = small
    cv2.rectangle(image, (x0 - 1, y0 - 1), (x0 + new_w, y0 + new_h), (255, 255, 255), 1)
    return image
