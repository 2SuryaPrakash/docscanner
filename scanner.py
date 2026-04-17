

import cv2
import itertools
import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple


def order_points(pts: np.ndarray) -> np.ndarray:
    pts  = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def fullimage_corners(h: int, w: int) -> np.ndarray:
    return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)


def is_degenerate_quad(corners: Optional[np.ndarray], threshold: int = 50) -> bool:
    if corners is None:
        return True
    pts = corners.reshape(4, 2)
    return all(
        np.linalg.norm(pts[i] - pts[j]) <= threshold
        for i in range(4) for j in range(i + 1, 4)
    )


def resize_for_processing(image: np.ndarray, max_edge: int = 1080
                          ) -> Tuple[np.ndarray, float]:
    h, w  = image.shape[:2]
    scale = min(max_edge / max(h, w), 1.0)
    if scale == 1.0:
        return image.copy(), 1.0
    return cv2.resize(image, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA), scale


def apply_clahe_normalisation(image: np.ndarray) -> np.ndarray:
    lab     = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def normalize_illumination(image: np.ndarray) -> np.ndarray:
    """
    Flatten shadows and uneven lighting by dividing by a background estimate.
    Steps: dilate to remove text - heavy median blur - divide original by
    this smooth background - stretch contrast.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return norm


def shadow_remove_for_segmentation(image: np.ndarray) -> np.ndarray:
    """
    Produce a shadow-free, text-suppressed image ideal for page segmentation.

    Pipeline:
    1. Convert to grayscale
    2. Heavy dilation (large kernel) to erase text strokes
    3. Large median blur to estimate smooth background illumination
    4. Divide original by background → flat illumination
    5. Strong bilateral filter to suppress residual text texture
    6. Return as BGR (so downstream functions that expect BGR still work)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Estimate background illumination
    # Large dilation removes text (thin dark strokes become background)
    dilated = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    # Heavy median blur smooths the dilated result into a background model
    bg = cv2.medianBlur(dilated, 51)

    # Step 2: Divide original by background to flatten illumination
    norm = cv2.divide(gray, bg, scale=255)

    # Step 3: Strong bilateral filter to suppress remaining text texture
    # while preserving the page-to-background edge
    smooth = cv2.bilateralFilter(norm, d=11, sigmaColor=100, sigmaSpace=100)

    return cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# LSD-based corner detection (shadow-robust, gap-free)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_corners(corners: List[Tuple[int, int]],
                    min_dist: int = 20) -> List[Tuple[int, int]]:
    """Remove corners that are within min_dist of an already-accepted one."""
    filtered: List[Tuple[int, int]] = []
    for c in corners:
        if all(np.hypot(c[0] - f[0], c[1] - f[1]) >= min_dist
               for f in filtered):
            filtered.append(c)
    return filtered


def _angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
    cos_a = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _angle_range_of_quad(quad: np.ndarray) -> float:
    """Return the range (max - min) of interior angles of a quadrilateral."""
    pts = quad.reshape(4, 2)
    angles = []
    for i in range(4):
        p0 = pts[(i - 1) % 4]
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        angles.append(_angle_between_vectors(p0 - p1, p2 - p1))
    return float(np.ptp(angles))


def _get_lsd_corners(gray_img: np.ndarray) -> List[Tuple[int, int]]:
    """
    Use OpenCV's Line Segment Detector (LSD) on raw grayscale to find
    candidate document corners.

    LSD detects line segments from gradient analysis — fundamentally
    different from Hough on Canny edges. It produces fewer, higher-quality
    segments that correspond to real structural edges (page boundaries)
    rather than text.

    Pipeline:
    1. LSD on preprocessed grayscale - line segments
    2. Separate into horizontal and vertical
    3. Draw each group onto a canvas, slightly extended
    4. Connected-components merge colinear segments
    5. Top-2 per orientation - endpoints = candidate corners
    6. Overlap of H and V canvases → additional intersection corners
    """
    h, w = gray_img.shape[:2]

    # --- Detect line segments with LSD on raw grayscale ----------------
    lsd = cv2.createLineSegmentDetector(0)  # LSD_REFINE_STD
    lines_raw, _, _, _ = lsd.detect(gray_img)

    if lines_raw is None or len(lines_raw) == 0:
        return []

    # LSD output: (N, 1, 4) → list of (x1, y1, x2, y2) tuples
    lines_list = []
    for seg in lines_raw:
        x1, y1, x2, y2 = seg[0]
        lines_list.append((int(x1), int(y1), int(x2), int(y2)))

    corners: List[Tuple[int, int]] = []
    horizontal_canvas = np.zeros((h, w), dtype=np.uint8)
    vertical_canvas   = np.zeros((h, w), dtype=np.uint8)

    for x1, y1, x2, y2 in lines_list:
        if abs(x2 - x1) > abs(y2 - y1):
            # horizontal — sort by x, extend slightly
            (x1, y1), (x2, y2) = sorted([(x1, y1), (x2, y2)],
                                         key=lambda pt: pt[0])
            cv2.line(horizontal_canvas,
                     (max(x1 - 5, 0), y1),
                     (min(x2 + 5, w - 1), y2), 255, 2)
        else:
            # vertical — sort by y, extend slightly
            (x1, y1), (x2, y2) = sorted([(x1, y1), (x2, y2)],
                                         key=lambda pt: pt[1])
            cv2.line(vertical_canvas,
                     (x1, max(y1 - 5, 0)),
                     (x2, min(y2 + 5, h - 1)), 255, 2)

    merged_lines: List[Tuple[int, int, int, int]] = []

    # Merge horizontal segments via connected components
    cnts_h, _ = cv2.findContours(horizontal_canvas, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
    cnts_h = sorted(cnts_h, key=lambda c: cv2.arcLength(c, True),
                    reverse=True)[:2]
    horizontal_canvas = np.zeros((h, w), dtype=np.uint8)
    for contour in cnts_h:
        pts = contour.reshape(-1, 2)
        min_x = int(np.amin(pts[:, 0])) + 2
        max_x = int(np.amax(pts[:, 0])) - 2
        left_y  = int(np.mean(pts[pts[:, 0] == np.amin(pts[:, 0])][:, 1]))
        right_y = int(np.mean(pts[pts[:, 0] == np.amax(pts[:, 0])][:, 1]))
        merged_lines.append((min_x, left_y, max_x, right_y))
        cv2.line(horizontal_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
        corners.append((min_x, left_y))
        corners.append((max_x, right_y))

    # Merge vertical segments via connected components
    cnts_v, _ = cv2.findContours(vertical_canvas, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
    cnts_v = sorted(cnts_v, key=lambda c: cv2.arcLength(c, True),
                    reverse=True)[:2]
    vertical_canvas = np.zeros((h, w), dtype=np.uint8)
    for contour in cnts_v:
        pts = contour.reshape(-1, 2)
        min_y = int(np.amin(pts[:, 1])) + 2
        max_y = int(np.amax(pts[:, 1])) - 2
        top_x    = int(np.mean(pts[pts[:, 1] == np.amin(pts[:, 1])][:, 0]))
        bottom_x = int(np.mean(pts[pts[:, 1] == np.amax(pts[:, 1])][:, 0]))
        merged_lines.append((top_x, min_y, bottom_x, max_y))
        cv2.line(vertical_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
        corners.append((top_x, min_y))
        corners.append((bottom_x, max_y))

    # Where horizontal and vertical lines overlap - additional corners
    overlap_y, overlap_x = np.where(
        (horizontal_canvas + vertical_canvas) == 2)
    corners += list(zip(overlap_x.tolist(), overlap_y.tolist()))

    corners = _filter_corners(corners)
    return corners


def find_corners_lsd(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect document corners using LSD (Line Segment Detector).
    """
    h, w = image.shape[:2]
    image_area = float(h * w)

    MIN_QUAD_AREA_RATIO = 0.25
    MAX_QUAD_ANGLE_RANGE = 40

    # Preprocessing matching the reference exactly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Morphological close: dilate then erode to bridge gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # LSD on the preprocessed grayscale (NOT on Canny edges!)
    test_corners = _get_lsd_corners(closed)

    approx_contours: List[np.ndarray] = []

    if len(test_corners) >= 4:
        quads = []
        for combo in itertools.combinations(test_corners, 4):
            pts = np.array(combo, dtype=np.float32)
            pts = order_points(pts)
            quads.append(pts.reshape(4, 1, 2).astype(np.int32))

        # Sort by area descending, take top 5
        quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
        # Then sort by angle range ascending (most rectangular first)
        quads = sorted(quads, key=_angle_range_of_quad)

        for approx in quads:
            cnt_area = cv2.contourArea(approx)
            if (len(approx) == 4
                    and cnt_area > image_area * MIN_QUAD_AREA_RATIO
                    and _angle_range_of_quad(approx) < MAX_QUAD_ANGLE_RANGE):
                approx_contours.append(approx)
                break

    # Also try direct contour detection from Canny edge map
    edged = cv2.Canny(closed, 0, 84)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        approx = cv2.approxPolyDP(c, 80, True)
        if (len(approx) == 4
                and cv2.contourArea(approx) > image_area * MIN_QUAD_AREA_RATIO
                and _angle_range_of_quad(approx) < MAX_QUAD_ANGLE_RANGE):
            approx_contours.append(approx)
            break

    if not approx_contours:
        return None

    best = max(approx_contours, key=cv2.contourArea)
    corners = best.reshape(4, 2).astype(np.float32)

    # Clamp to image bounds
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)

    return order_points(corners)



def _otsu_canny_thresh(gray: np.ndarray) -> Tuple[int, int]:
    med = float(np.median(gray))
    med = max(40.0, min(210.0, med))
    return int(0.66 * med), int(1.33 * med)


def detect_edges_canny(image: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filt  = cv2.bilateralFilter(gray, d=9, sigmaColor=50, sigmaSpace=50)
    lo, hi = _otsu_canny_thresh(filt)
    edges  = cv2.Canny(filt, lo, hi)
    k      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(edges, k, iterations=2)


def detect_edges_color_seg(image: np.ndarray) -> np.ndarray:
    h, w    = image.shape[:2]
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    mask  = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    for corner in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
        cv2.floodFill(blurred, mask, corner, 255, loDiff=15, upDiff=15, flags=flags)
    doc = cv2.bitwise_not(mask[1:-1, 1:-1])
    k   = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    doc = cv2.morphologyEx(doc, cv2.MORPH_CLOSE, k)
    edges = cv2.Canny(doc, 30, 120)
    k2    = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.dilate(edges, k2, iterations=2)


def detect_edges_morph_gradient(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blr  = cv2.GaussianBlur(gray, (5, 5), 0)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    grad = cv2.morphologyEx(blr, cv2.MORPH_GRADIENT, k)
    _, edges = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k2   = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k2)


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation-based document detection 
# ─────────────────────────────────────────────────────────────────────────────

def segment_document_closing(image: np.ndarray
                             ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    
    h, w = image.shape[:2]
    image_area = float(h * w)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Input is already shadow-free & text-suppressed, light smoothing suffices
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edges with broad thresholds to catch the page boundary
    med = float(np.median(blurred))
    lo = int(max(30, 0.5 * med))
    hi = int(min(250, 1.5 * med))
    edges = cv2.Canny(blurred, lo, hi)

    # Moderate dilation to bridge small gaps between text edges
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edges, k_dilate, iterations=2)

    # Reduced closing kernel to avoid excessive boundary erosion
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, k_close)

    # Fill interior holes: flood-fill from (0,0) to find background,
    # then invert to get the filled page
    flood = closed.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    filled = cv2.bitwise_or(closed, cv2.bitwise_not(flood))

    # Smaller opening to preserve boundary accuracy
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    cleaned = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k_open)

    # Find the largest connected component (= the document page)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    if num_labels < 2:
        return None, cleaned

    # Skip label 0 (background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    corners = _quad_from_mask(mask, image_area)
    return corners, mask


def segment_document_floodfill(image: np.ndarray
                               ) -> Tuple[Optional[np.ndarray], np.ndarray]:
   
    h, w = image.shape[:2]
    image_area = float(h * w)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Input is already shadow-free & text-suppressed, light smoothing suffices
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Flood-fill from many seed points along all 4 edges of the image
    # (not just 4 corners) for robust background coverage
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    step = max(20, min(h, w) // 20)  # ~20 seeds per edge

    seeds = []
    for x in range(0, w, step):
        seeds.extend([(x, 0), (x, h - 1)])
    for y in range(0, h, step):
        seeds.extend([(0, y), (w - 1, y)])
    # Always include actual corners
    seeds.extend([(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)])

    for seed in seeds:
        cv2.floodFill(blurred, flood_mask, seed, 255,
                      loDiff=25, upDiff=25, flags=flags)

    # Invert: background = 0, document = 255
    doc_mask = cv2.bitwise_not(flood_mask[1:-1, 1:-1])

    # Large closing to solidify the page mask
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    doc_mask = cv2.morphologyEx(doc_mask, cv2.MORPH_CLOSE, k_close)

    # Opening to clean small noise
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    doc_mask = cv2.morphologyEx(doc_mask, cv2.MORPH_OPEN, k_open)

    corners = _quad_from_mask(doc_mask, image_area)
    return corners, doc_mask


def _quad_from_mask(mask: np.ndarray, image_area: float
                    ) -> Optional[np.ndarray]:
    """
    Given a binary mask, find the largest contour, approximate it to a
    quadrilateral, and validate it.

    Slight mask dilation before contour extraction compensates for
    boundary erosion caused by morphological closing/opening upstream.
    """
    # Dilate the mask slightly to push boundary outward (undo erosion)
    k_expand = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    expanded = cv2.dilate(mask, k_expand, iterations=1)

    cnts, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Sort by area descending, try the top candidates
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for cnt in cnts[:5]:
        area = cv2.contourArea(cnt)
        if area < 0.05 * image_area:
            break

        # Use convex hull for tighter outer boundary
        hull = cv2.convexHull(cnt)
        pts = _approx_to_quad(hull, image_area)
        if pts is not None:
            return order_points(pts)

        # Try the raw contour too
        pts = _approx_to_quad(cnt, image_area)
        if pts is not None:
            return order_points(pts)

        # Fallback: min-area bounding rectangle
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect).astype(np.float32)
        if _validate_quad(box, image_area, min_area_frac=0.10):
            return order_points(box)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Quad validation & scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _largest_cnt_area(edges: np.ndarray) -> float:
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    return float(cv2.contourArea(max(cnts, key=cv2.contourArea)))


def _poly_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _quad_rectangularity(pts: np.ndarray) -> float:
    score = 0.0
    for i in range(4):
        p0, p1, p2 = pts[i], pts[(i+1) % 4], pts[(i+2) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_a = abs(float(np.dot(v1, v2) / (n1 * n2)))
        score += 1.0 - cos_a
    return score / 4.0


def _min_interior_angle_deg(pts: np.ndarray) -> float:
    min_angle = 180.0
    for i in range(4):
        p0 = pts[(i - 1) % 4]
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        v1 = p0 - p1
        v2 = p2 - p1
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_a = np.dot(v1, v2) / (n1 * n2)
        angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        min_angle = min(min_angle, angle)
    return min_angle


def _validate_quad(pts: np.ndarray, image_area: float,
                   min_area_frac: float = 0.15) -> bool:
    if pts is None or len(pts) != 4:
        return False
    ordered = order_points(pts)

    if _poly_area(ordered) < min_area_frac * image_area:
        return False

    sides = [np.linalg.norm(ordered[(i+1) % 4] - ordered[i]) for i in range(4)]
    if min(sides) < 5.0:
        return False
    if max(sides) / (min(sides) + 1e-6) > 20.0:
        return False

    if _min_interior_angle_deg(ordered) < 25.0:
        return False

    return True


def _score_quad(pts: np.ndarray, image_area: float) -> float:
    ordered   = order_points(pts)
    area_frac = min(_poly_area(ordered) / image_area, 0.90) / 0.90
    rect      = _quad_rectangularity(ordered)
    return 0.80 * area_frac + 0.20 * rect


def _approx_to_quad(contour: np.ndarray,
                    image_area: float) -> Optional[np.ndarray]:
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, closed=True)
    if peri < 20:
        return None
    for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                0.10, 0.12, 0.15, 0.18, 0.20]:
        approx = cv2.approxPolyDP(hull, eps * peri, closed=True)
        n = len(approx)
        if n == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            if _validate_quad(pts, image_area):
                return pts
        if n < 4:
            break
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Unified detection flow: segmentation-first, then edge fallback
# ─────────────────────────────────────────────────────────────────────────────

def detect_edges(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """ edge detection for find_document_quad."""
    h, w     = image.shape[:2]
    img_area = float(h * w)
    e_canny = detect_edges_canny(image)
    a_canny = _largest_cnt_area(e_canny) / img_area
    if a_canny >= 0.15:
        return e_canny, "bilateral-canny"
    e_seg   = detect_edges_color_seg(image)
    a_seg   = _largest_cnt_area(e_seg) / img_area
    e_morph = detect_edges_morph_gradient(image)
    a_morph = _largest_cnt_area(e_morph) / img_area
    best = max(a_canny, a_seg, a_morph)
    if best == a_seg:
        return e_seg, "color-seg"
    if best == a_morph:
        return e_morph, "morph-gradient"
    return e_canny, "bilateral-canny"


def find_document_corners_edge_fallback(
    edges: np.ndarray, image_shape: Tuple
) -> Optional[np.ndarray]:
    """Original edge-based corner finder — used as third-tier fallback."""
    h, w       = image_shape[:2]
    image_area = float(h * w)

    k      = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    all_cnts: List[np.ndarray] = []
    for mode in [cv2.RETR_EXTERNAL, cv2.RETR_LIST]:
        cnts, _ = cv2.findContours(closed, mode, cv2.CHAIN_APPROX_SIMPLE)
        all_cnts.extend(cnts)

    if not all_cnts:
        return None

    all_cnts.sort(key=cv2.contourArea, reverse=True)
    seen: set = set()
    unique: List[np.ndarray] = []
    for c in all_cnts:
        cid = id(c)
        if cid not in seen:
            seen.add(cid)
            unique.append(c)

    candidates: List[Tuple[float, np.ndarray]] = []
    fallback_cnt: Optional[np.ndarray] = None

    for cnt in unique[:20]:
        area = cv2.contourArea(cnt)
        if area < 0.03 * image_area:
            break

        if fallback_cnt is None:
            fallback_cnt = cnt

        pts = _approx_to_quad(cnt, image_area)
        if pts is not None:
            candidates.append((_score_quad(pts, image_area), pts))

    if candidates:
        _, best = max(candidates, key=lambda x: x[0])
        return order_points(best)

    if fallback_cnt is None:
        return None

    rect = cv2.minAreaRect(fallback_cnt)
    box  = cv2.boxPoints(rect).astype(np.float32)
    if _validate_quad(box, image_area, min_area_frac=0.05):
        return order_points(box)

    pts_all = fallback_cnt.reshape(-1, 2).astype(np.float32)
    extreme = np.array([
        pts_all[np.argmin(pts_all[:, 1])],
        pts_all[np.argmax(pts_all[:, 0])],
        pts_all[np.argmax(pts_all[:, 1])],
        pts_all[np.argmin(pts_all[:, 0])],
    ], dtype=np.float32)
    if _validate_quad(extreme, image_area, min_area_frac=0.05):
        return order_points(extreme)

    return None


def find_document_quad(
    image: np.ndarray,
) -> Tuple[Optional[np.ndarray], np.ndarray, str]:
    """
    Unified detection: tries Hough-line intersection first (shadow-robust,
    gap-free), then segmentation methods, then edge fallback.

    Returns:
        (corners_or_None, debug_mask, method_string)
    """
    h, w = image.shape[:2]
    empty_mask = np.zeros((h, w), dtype=np.uint8)

    # ── Approach 0: LSD line-segment corner detection (shadow-robust) ─────
    try:
        corners_h = find_corners_lsd(image)
        if corners_h is not None:
            # Build a debug mask showing the detected quad
            lsd_mask = np.zeros((h, w), dtype=np.uint8)
            pts = corners_h.reshape(4, 2).astype(np.int32)
            cv2.fillPoly(lsd_mask, [pts], 255)
            return corners_h, lsd_mask, "lsd-lines"
    except Exception:
        pass

    # ── Preprocess for segmentation: shadow removal + text suppression ─────
    shadow_free = shadow_remove_for_segmentation(image)

    # ── Approach A: Closing + largest connected component ───────────────────
    try:
        corners_a, mask_a = segment_document_closing(shadow_free)
        if corners_a is not None:
            return corners_a, mask_a, "seg-closing"
    except Exception:
        mask_a = empty_mask

    # ── Approach C: Flood-fill background removal ──────────────────────────
    try:
        corners_c, mask_c = segment_document_floodfill(shadow_free)
        if corners_c is not None:
            return corners_c, mask_c, "seg-floodfill"
    except Exception:
        mask_c = empty_mask

    # ── Edge-based fallback ────────────────────────────────────────────────
    try:
        edges, edge_method = detect_edges(image)
        corners_e = find_document_corners_edge_fallback(edges, image.shape)
        if corners_e is not None:
            return corners_e, edges, f"edge-{edge_method}"
        # Return edges as debug mask even if no quad found
        return None, edges, f"edge-{edge_method} (no quad)"
    except Exception:
        return None, empty_mask, "failed"


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    tl, tr, br, bl = rect
    max_w = max(int(np.linalg.norm(tr - tl)), int(np.linalg.norm(br - bl)))
    max_h = max(int(np.linalg.norm(bl - tl)), int(np.linalg.norm(br - tr)))
    if max_w < 100 or max_h < 100:
        raise ValueError(f"Warp output too small: {max_w}x{max_h}")
    dst = np.array([[0, 0], [max_w-1, 0], [max_w-1, max_h-1], [0, max_h-1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_w, max_h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def _filter_original(img: np.ndarray) -> np.ndarray:
    return img.copy()


def _filter_magic_colour(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b, g, r = cv2.split(img)
    enhanced = cv2.merge([clahe.apply(b), clahe.apply(g), clahe.apply(r)])
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * (1.0 + (1.0 - s / 255.0) * 0.5), 0, 255)
    return cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2BGR)


def _filter_bw(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw   = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, blockSize=21, C=10)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def _filter_grayscale(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX),
                        cv2.COLOR_GRAY2BGR)


def _filter_pencil(img: np.ndarray) -> np.ndarray:
    gray_sketch, _ = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2BGR)


def _filter_shadow(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k    = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    bg   = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    norm = np.clip(gray.astype(np.float64) / (bg.astype(np.float64) + 1e-6) * 255,
                   0, 255).astype(np.uint8)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


FILTERS: Dict[str, Any] = {
    "Original":            _filter_original,
    "Magic Colour":        _filter_magic_colour,
    "Black & White":       _filter_bw,
    "Grayscale":           _filter_grayscale,
    "Pencil Sketch":       _filter_pencil,
    "Hard Shadow Removal": _filter_shadow,
}


def downsample_for_display(image: np.ndarray, max_width: int = 600) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy()
    s = max_width / w
    return cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def draw_quad_on_image(image: np.ndarray, corners: np.ndarray,
                       display_scale: float = 1.0) -> np.ndarray:
    vis  = image.copy()
    pts  = (corners.reshape(4, 2) * display_scale).astype(np.int32)
    cv2.polylines(vis, [pts], isClosed=True, color=(30, 220, 80), thickness=3)
    colors = [(30, 30, 220), (220, 80, 0), (0, 200, 200), (180, 0, 200)]
    labels = ["TL", "TR", "BR", "BL"]
    for pt, col, lbl in zip(pts, colors, labels):
        cv2.circle(vis, tuple(pt), 12, col, -1)
        cv2.circle(vis, tuple(pt), 12, (255, 255, 255), 2)
        cv2.putText(vis, lbl, (int(pt[0]) + 15, int(pt[1]) + 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, col, 2, cv2.LINE_AA)
    return vis


def make_lsd_debug_vis(image: np.ndarray) -> np.ndarray:
    """
    Draw LSD-detected line segments on a copy of *image*.
    Uses the same preprocessing as find_corners_lsd so the segments match
    what the detector actually sees.
    """
    vis  = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    lsd = cv2.createLineSegmentDetector(0)
    lines_raw, _, _, _ = lsd.detect(closed)
    if lines_raw is None or len(lines_raw) == 0:
        return vis
    for seg in lines_raw:
        x1, y1, x2, y2 = (int(v) for v in seg[0])
        color = (220, 100, 30) if abs(x2 - x1) >= abs(y2 - y1) else (30, 80, 220)
        cv2.line(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    return vis


def run_pipeline(
    image_bytes: bytes,
    filter_name: str = "Original",
    manual_corners: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    End-to-end pipeline.  ALL four detection tiers always run so that
    their intermediate images can be displayed in the UI.
    stages["winning_tier"] records which tier produced the final corners.
    """
    stages: Dict[str, Any] = {}
    timings: Dict[str, float] = {}
    ok: Dict[str, bool] = {}

    t = time.perf_counter()
    try:
        arr      = np.frombuffer(image_bytes, dtype=np.uint8)
        original = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if original is None:
            return {"error": "Could not decode image."}
        stages["original"] = original
        ok["decode"] = True
    except Exception as exc:
        return {"error": f"Decode error: {exc}"}
    timings["decode"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    try:
        resized, scale = resize_for_processing(original, max_edge=1080)
        ok["resize"] = True
    except Exception:
        resized, scale = original.copy(), 1.0
        ok["resize"] = False
    stages["resized"] = resized
    stages["scale"]   = scale
    timings["resize"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    try:
        clahe_img = apply_clahe_normalisation(resized)
        ok["clahe"] = True
    except Exception:
        clahe_img = resized.copy()
        ok["clahe"] = False
    stages["clahe"] = clahe_img
    timings["clahe"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    try:
        shadow_free = shadow_remove_for_segmentation(clahe_img)
        ok["shadow_remove"] = True
    except Exception:
        shadow_free = clahe_img.copy()
        ok["shadow_remove"] = False
    stages["shadow_free"] = shadow_free
    timings["shadow_remove"] = (time.perf_counter() - t) * 1000

    # ── Detection: run ALL tiers, capture every intermediate ──────────────
    ph, pw     = resized.shape[:2]
    empty_mask = np.zeros((ph, pw), dtype=np.uint8)
    corners_proc  = None
    corner_method = ""
    winning_tier  = "none"

    t0_detect = time.perf_counter()

    if manual_corners is not None:
        corners_proc  = order_points(manual_corners.astype(np.float32))
        corner_method = "manual"
        winning_tier  = "manual"
        ok["detection"] = True
        stages["lsd_vis"]        = make_lsd_debug_vis(clahe_img)
        stages["lsd_mask"]       = empty_mask
        stages["closing_mask"]   = empty_mask
        stages["floodfill_mask"] = empty_mask
        stages["edge_map"]       = empty_mask
    else:
        # ── Tier 1: LSD line-segment detector ────────────────────────────
        stages["lsd_vis"] = make_lsd_debug_vis(clahe_img)
        try:
            lsd_corners = find_corners_lsd(clahe_img)
            stages["lsd_corners"] = lsd_corners   # None if failed
            if lsd_corners is not None:
                lsd_mask = np.zeros((ph, pw), dtype=np.uint8)
                pts = lsd_corners.reshape(4, 2).astype(np.int32)
                cv2.fillPoly(lsd_mask, [pts], 255)
                stages["lsd_mask"] = lsd_mask
                corners_proc  = lsd_corners
                corner_method = "lsd-lines"
                winning_tier  = "lsd"
            else:
                stages["lsd_mask"] = empty_mask
        except Exception:
            stages["lsd_mask"]    = empty_mask
            stages["lsd_corners"] = None

        # ── Tier 2: Segmentation — morphological closing ──────────────────
        # shadow_free is the correct input for segmentation tiers
        try:
            corners_a, mask_a = segment_document_closing(shadow_free)
            stages["closing_mask"]    = mask_a
            stages["closing_corners"] = corners_a   # None if failed
            if corners_proc is None and corners_a is not None:
                corners_proc  = corners_a
                corner_method = "seg-closing"
                winning_tier  = "seg-closing"
        except Exception:
            stages["closing_mask"]    = empty_mask
            stages["closing_corners"] = None

        # ── Tier 3: Segmentation — flood-fill background removal ──────────
        try:
            corners_c, mask_c = segment_document_floodfill(shadow_free)
            stages["floodfill_mask"]    = mask_c
            stages["floodfill_corners"] = corners_c   # None if failed
            if corners_proc is None and corners_c is not None:
                corners_proc  = corners_c
                corner_method = "seg-floodfill"
                winning_tier  = "seg-floodfill"
        except Exception:
            stages["floodfill_mask"]    = empty_mask
            stages["floodfill_corners"] = None

        # ── Tier 4: Edge-based fallback ───────────────────────────────────
        try:
            edges, edge_method = detect_edges(clahe_img)
            stages["edge_map"] = edges
            corners_e: Optional[np.ndarray] = None
            if corners_proc is None:
                corners_e = find_document_corners_edge_fallback(edges, resized.shape)
                if corners_e is not None:
                    corners_proc  = corners_e
                    corner_method = f"edge-{edge_method}"
                    winning_tier  = "edge"
            else:
                # Still detect for the override option even if not needed for auto
                corners_e = find_document_corners_edge_fallback(edges, resized.shape)
            stages["edge_corners"] = corners_e
        except Exception:
            stages["edge_map"]     = empty_mask
            stages["edge_corners"] = None

        # ── Final fallback: whole-image corners ───────────────────────────
        if corners_proc is None or is_degenerate_quad(corners_proc, threshold=50):
            corners_proc   = fullimage_corners(ph, pw)
            corner_method  = (corner_method + "  full-image fallback"
                              if corner_method else "full-image fallback")
            winning_tier   = "none"
            ok["detection"] = False
        else:
            ok["detection"] = True

    timings["detection"] = (time.perf_counter() - t0_detect) * 1000

    stages["corners_proc"]  = corners_proc
    stages["corner_method"] = corner_method
    stages["winning_tier"]  = winning_tier

    try:
        stages["contour_vis"]    = draw_quad_on_image(resized.copy(), corners_proc)
        stages["corner_overlay"] = draw_quad_on_image(resized.copy(), corners_proc)
    except Exception:
        stages["contour_vis"] = stages["corner_overlay"] = resized.copy()

    t = time.perf_counter()
    try:
        warped = four_point_transform(original, corners_proc / scale)
        ok["warp"] = True
    except Exception:
        warped = original.copy()
        ok["warp"] = False
    stages["warped"] = warped
    timings["warp"]  = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    try:
        enhanced = FILTERS.get(filter_name, _filter_original)(warped)
        ok["enhance"] = True
    except Exception:
        enhanced = warped.copy()
        ok["enhance"] = False
    stages["enhanced"] = enhanced
    timings["enhance"] = (time.perf_counter() - t) * 1000

    stages["ok"]         = ok
    stages["timings"]    = timings
    stages["total_time"] = sum(timings.values())
    return stages