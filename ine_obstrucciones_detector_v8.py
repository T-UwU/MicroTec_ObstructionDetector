import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configure logging for threshold tuning
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Canonical card size for warped view (INE aspect ratio ~1.586)
WARP_W = 1000
WARP_H = 630

# ROI definitions normalized [0,1] relative to warped image
# Format: (x0, y0, x1, y1) as fractions
FRONT_ROIS = {
    "main_photo": (0.06, 0.26, 0.30, 0.72),       # Primary photo - used for face detection
    "hologram_photo": (0.74, 0.26, 0.96, 0.65),   # Small hologram photo on right
    "text_center": (0.30, 0.20, 0.74, 0.58),      # Name and details area
    "text_bottom": (0.30, 0.58, 0.74, 0.96),      # Address and other info
    "lower_left": (0.05, 0.62, 0.30, 0.96),       # Signature area
}

BACK_ROIS = {
    "qr_left": (0.05, 0.15, 0.34, 0.68),          # Large QR code left
    "qr_mid": (0.34, 0.15, 0.63, 0.68),           # Large QR code middle
    "qr_small": (0.64, 0.20, 0.83, 0.54),         # Small QR code right
    "mrz": (0.05, 0.70, 0.96, 0.98),              # Machine Readable Zone (3 lines)
}

# Detection thresholds - tunable per ROI type
THRESHOLDS = {
    # Front side thresholds
    "front": {
        "main_photo": {
            "min_edge_density": 0.04,      # Photos have moderate edges
            "blob_v_perc": 72,             # Brightness percentile (stricter)
            "blob_mag_perc": 48,           # Gradient magnitude percentile
            "blob_min_area": 0.08,         # Min blob area fraction
            "blob_min_rect": 0.42,         # Rectangularity threshold
            "color_std_max": 25,           # Stricter uniformity (was 35)
            "require_low_std_or_high_sat": True,  # Special flag for main_photo
        },
        "hologram_photo": {
            "min_edge_density": 0.020,     # Hologram can be faint
            "blob_v_perc": 70,             
            "blob_mag_perc": 48,           
            "blob_min_area": 0.07,         # Lower min area
            "blob_min_rect": 0.55,         # Stricter rect (was 0.45)
            "color_std_max": 15,           # MUCH stricter (was 40) - key to filter FPs
        },
        "text_center": {
            "min_edge_density": 0.03,      # Text areas have some edges
            "blob_v_perc": 75,             # Higher threshold to reduce FPs
            "blob_mag_perc": 52,
            "blob_min_area": 0.15,         # Require larger blobs
            "blob_min_rect": 0.72,         # MUCH stricter (was 0.60)
            "color_std_max": 25,           # Stricter uniformity
        },
        "text_bottom": {
            "min_edge_density": 0.04,
            "blob_v_perc": 68,             # V8: More sensitive (was 70)
            "blob_mag_perc": 48,           # V8: More sensitive (was 50)
            "blob_min_area": 0.08,         # V8: Reduced (was 0.10)
            "blob_min_rect": 0.50,         # V8: Reduced (was 0.65)
            "color_std_max": 25,           # V8: Relaxed (was 20)
        },
        "lower_left": {
            "min_edge_density": 0.03,      # Signature area can be sparse
            "blob_v_perc": 68,             # V8: More sensitive (was 70)
            "blob_mag_perc": 50,           # V8: More sensitive (was 52)
            "blob_min_area": 0.12,         # V8: Reduced (was 0.15)
            "blob_min_rect": 0.35,         # V8: MUCH reduced (was 0.55) - papers aren't always perfectly rectangular
            "color_std_max": 15,           # V8: Relaxed (was 10) - allow more variation
        },
    },
    # Back side thresholds
    "back": {
        "qr_left": {
            "min_edge_density": 0.10,      # QR codes have high edge density
            "blob_v_perc": 72,             # Higher (was 68)
            "blob_mag_perc": 55,
            "blob_min_area": 0.12,         # Higher (was 0.10)
            "blob_min_rect": 0.55,         # Higher (was 0.50)
            "color_std_max": 12,           # MUCH stricter (was 30)
            "min_black_ratio": 0.15,       # QR has significant black areas
        },
        "qr_mid": {
            "min_edge_density": 0.10,
            "blob_v_perc": 72,             # Higher
            "blob_mag_perc": 55,
            "blob_min_area": 0.12,         # Higher
            "blob_min_rect": 0.55,         # Higher
            "color_std_max": 12,           # MUCH stricter
            "min_black_ratio": 0.15,
        },
        "qr_small": {
            "min_edge_density": 0.06,      # Smaller QR, lower threshold
            "blob_v_perc": 72,             # Higher (was 70)
            "blob_mag_perc": 52,
            "blob_min_area": 0.08,
            "blob_min_rect": 0.58,         # Higher (was 0.55)
            "color_std_max": 15,           # MUCH stricter (was 35)
            "min_black_ratio": 0.10,
        },
        "mrz": {
            "min_edge_density": 0.025,     # V8: Lower (was 0.03) - better detection of covered MRZ
            "blob_v_perc": 66,             # V8: More sensitive (was 68)
            "blob_mag_perc": 50,           # V8: More sensitive (was 55)
            "blob_min_area": 0.08,         # V8: Reduced (was 0.12) - catch smaller obstructions
            "blob_min_rect": 0.45,         # V8: Reduced (was 0.60)
            "color_std_max": 20,           # V8: Relaxed (was 15)
        },
    },
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WarpResult:
    """Result of perspective warp operation."""
    warped: np.ndarray
    M: np.ndarray  # Transform matrix: original -> warped
    quad: np.ndarray  # Detected card corners in original image


@dataclass
class BlobInfo:
    """Information about a detected bright flat blob."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area_frac: float
    rectangularity: float
    color_std: float  # Color standard deviation (uniformity)
    mean_saturation: float
    edge_sharpness: float  # Higher = sharper edges (paper), lower = gradient (glare)
    v_thr: float
    m_thr: float


@dataclass 
class ObstructionResult:
    """Result of obstruction detection for a single ROI."""
    obstructed: bool
    confidence: float  # 0-1, higher = more confident it's an obstruction
    reason: str  # Human-readable explanation
    edge_density: float
    blob: Optional[BlobInfo] = None
    black_ratio: Optional[float] = None  # For QR regions
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SideDetectionResult:
    """Result of side detection."""
    side: str  # "front" or "back"
    confidence: float
    face_detected: bool
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    qr_signature_score: float = 0.0
    mrz_lines: int = 0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points as: top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]      # Top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]      # Bottom-right: largest x+y
    rect[1] = pts[np.argmin(diff)]   # Top-right: smallest x-y
    rect[3] = pts[np.argmax(diff)]   # Bottom-left: largest x-y
    return rect


def compute_edge_density(bgr: np.ndarray, blur_size: int = 3) -> float:
    """Compute edge density (fraction of edge pixels)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    edges = cv2.Canny(gray, 50, 150)
    return float((edges > 0).mean())


def compute_black_ratio(bgr: np.ndarray, threshold: int = 80) -> float:
    """Compute ratio of dark/black pixels (for QR detection)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float((gray < threshold).mean())


def compute_local_variance(gray: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Compute local variance map for texture analysis."""
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (kernel_size, kernel_size))
    sqr_mean = cv2.blur(gray_f ** 2, (kernel_size, kernel_size))
    variance = sqr_mean - mean ** 2
    return np.maximum(variance, 0)


# =============================================================================
# ADVANCED TEXTURE AND FREQUENCY ANALYSIS (V7 NEW)
# =============================================================================

def compute_lbp_entropy(gray: np.ndarray, num_points: int = 24, radius: int = 3) -> float:
    """
    Compute Local Binary Pattern texture entropy.
    
    LBP characterizes local texture by comparing each pixel with its neighbors.
    Uniform obstructions (paper, sticky notes) have LOW entropy (<2.5)
    Natural document content (text, photos) has HIGH entropy (>3.5)
    
    This is a simplified LBP implementation that doesn't require skimage.
    
    Args:
        gray: Grayscale image
        num_points: Number of neighbors to sample (8, 16, or 24)
        radius: Radius of the circle for sampling neighbors
        
    Returns:
        Entropy value (0-8 typical range, lower = more uniform)
    """
    h, w = gray.shape
    if h < 2*radius + 1 or w < 2*radius + 1:
        return 4.0  # Default mid-range value for tiny regions
    
    # Simplified uniform LBP with 8 points
    # Sample 8 neighbors at the given radius
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    
    # Create coordinate offsets for neighbors
    dx = np.round(radius * np.cos(angles)).astype(int)
    dy = np.round(radius * np.sin(angles)).astype(int)
    
    # Compute LBP for each pixel (excluding border)
    lbp = np.zeros((h - 2*radius, w - 2*radius), dtype=np.uint8)
    center = gray[radius:h-radius, radius:w-radius].astype(np.float32)
    
    for i, (ddx, ddy) in enumerate(zip(dx, dy)):
        neighbor = gray[radius+ddy:h-radius+ddy, radius+ddx:w-radius+ddx].astype(np.float32)
        lbp += ((neighbor >= center).astype(np.uint8) << i)
    
    # Compute histogram and entropy
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-10)  # Normalize
    
    # Shannon entropy
    nonzero = hist > 0
    entropy = -np.sum(hist[nonzero] * np.log2(hist[nonzero]))
    
    return float(entropy)


def compute_dct_uniformity(gray: np.ndarray, block_size: int = 16) -> float:
    """
    Compute DCT-based uniformity score using DC/AC energy ratio.
    
    The Discrete Cosine Transform decomposes an image into frequency components.
    - DC coefficient (top-left) represents average brightness
    - AC coefficients represent variations/texture
    
    Uniform obstructions have HIGH DC ratio (>0.6) - almost all energy is in DC
    Natural content has LOW DC ratio (<0.3) - energy distributed across frequencies
    
    Args:
        gray: Grayscale image
        block_size: Size of DCT blocks (8, 16, or 32)
        
    Returns:
        DC energy ratio (0-1, higher = more uniform = suspicious)
    """
    h, w = gray.shape
    if h < block_size or w < block_size:
        return 0.5  # Default mid-range
    
    # Crop to multiple of block_size
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    gray_crop = gray[:h_crop, :w_crop].astype(np.float32)
    
    total_dc_energy = 0.0
    total_energy = 0.0
    num_blocks = 0
    
    for y in range(0, h_crop, block_size):
        for x in range(0, w_crop, block_size):
            block = gray_crop[y:y+block_size, x:x+block_size]
            
            # Apply DCT
            dct_block = cv2.dct(block)
            
            # DC energy is the square of the DC coefficient
            dc_energy = dct_block[0, 0] ** 2
            
            # Total energy is sum of squares of all coefficients
            block_energy = np.sum(dct_block ** 2)
            
            total_dc_energy += dc_energy
            total_energy += block_energy
            num_blocks += 1
    
    if total_energy < 1e-10:
        return 1.0  # Completely uniform (black or white)
    
    dc_ratio = total_dc_energy / total_energy
    return float(dc_ratio)


def classify_glare_vs_paper(bgr: np.ndarray, contour: np.ndarray) -> Dict[str, Any]:
    """
    Classify a bright region as glare or paper obstruction.
    
    Key discriminating features:
    - Paper has SHARP edges (high gradient at boundary, low inside)
    - Glare has SMOOTH transitions (gradual gradient, no sharp boundary)
    
    The edge-to-interior gradient ratio distinguishes them:
    - Paper: ratio > 3.0
    - Glare: ratio < 2.0
    
    Args:
        bgr: Color image
        contour: Contour of the bright region
        
    Returns:
        Dictionary with classification results
    """
    h, w = bgr.shape[:2]
    
    # Create edge mask (3px band around contour)
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(edge_mask, [contour], -1, 255, 4)  # 4px thick edge
    
    # Create interior mask (eroded region)
    interior_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(interior_mask, [contour], -1, 255, -1)  # Filled
    kernel = np.ones((9, 9), np.uint8)
    interior_mask = cv2.erode(interior_mask, kernel, iterations=1)
    
    # Remove edge from interior
    interior_mask = cv2.bitwise_and(interior_mask, cv2.bitwise_not(edge_mask))
    
    # Compute gradient magnitude
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    
    # Compute mean gradients
    edge_pixels = gradient[edge_mask > 0]
    interior_pixels = gradient[interior_mask > 0]
    
    if len(edge_pixels) == 0 or len(interior_pixels) == 0:
        return {'is_glare': False, 'is_paper': False, 'edge_ratio': 0, 'confidence': 0}
    
    edge_grad = float(np.mean(edge_pixels))
    interior_grad = float(np.mean(interior_pixels))
    
    # Compute ratio (avoid division by zero)
    ratio = edge_grad / (interior_grad + 1e-6)
    
    # HSV analysis for additional discrimination
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    region_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(region_mask, [contour], -1, 255, -1)
    
    mean_saturation = float(np.mean(hsv[:, :, 1][region_mask > 0]))
    mean_value = float(np.mean(hsv[:, :, 2][region_mask > 0]))
    
    # Classification logic
    # Glare: low saturation, very high value, smooth transition (low ratio)
    is_glare = ratio < 2.0 and mean_saturation < 35 and mean_value > 210
    
    # Paper: higher ratio (sharp edges), can have any saturation
    is_paper = ratio > 2.5 and mean_value > 150
    
    # Confidence based on how clear the classification is
    if is_glare:
        confidence = min(1.0, (2.0 - ratio) / 1.5)
    elif is_paper:
        confidence = min(1.0, (ratio - 2.5) / 2.0)
    else:
        confidence = 0.3  # Ambiguous
    
    return {
        'is_glare': is_glare,
        'is_paper': is_paper,
        'edge_ratio': ratio,
        'edge_grad': edge_grad,
        'interior_grad': interior_grad,
        'mean_saturation': mean_saturation,
        'mean_value': mean_value,
        'confidence': confidence
    }


def compute_obstruction_score(bgr: np.ndarray) -> Dict[str, float]:
    """
    Compute a multi-signal obstruction score combining all detection methods.
    
    This function aggregates:
    1. Local variance (existing method)
    2. LBP texture entropy (new)
    3. DCT frequency analysis (new)
    4. Edge density (existing method)
    
    Returns scores and a combined probability of obstruction.
    
    Args:
        bgr: Color image region to analyze
        
    Returns:
        Dictionary with individual scores and combined obstruction probability
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1. Local variance score (low variance = suspicious)
    local_var = np.mean(compute_local_variance(gray, kernel_size=11))
    # Normalize: <100 is very uniform, >500 is textured
    variance_score = 1.0 - np.clip(local_var / 400, 0, 1)
    
    # 2. LBP entropy score (low entropy = suspicious)
    lbp_entropy = compute_lbp_entropy(gray, radius=2)
    # Normalize: <2.5 is uniform, >4.0 is textured
    lbp_score = 1.0 - np.clip((lbp_entropy - 1.5) / 3.0, 0, 1)
    
    # 3. DCT uniformity score (high DC ratio = suspicious)
    dct_ratio = compute_dct_uniformity(gray, block_size=8)
    # Already normalized 0-1, higher = more uniform
    dct_score = dct_ratio
    
    # 4. Edge density score (low edges = suspicious)
    edge_density = compute_edge_density(bgr)
    # Normalize: <0.03 is very uniform, >0.10 is textured
    edge_score = 1.0 - np.clip(edge_density / 0.08, 0, 1)
    
    # 5. Color uniformity score
    color_std = float(np.std(bgr))
    # Normalize: <15 is very uniform, >40 is varied
    color_score = 1.0 - np.clip(color_std / 35, 0, 1)
    
    # Weighted combination
    # LBP and DCT are the strongest new signals
    weights = {
        'variance': 0.15,
        'lbp': 0.25,
        'dct': 0.25,
        'edge': 0.20,
        'color': 0.15
    }
    
    combined = (
        weights['variance'] * variance_score +
        weights['lbp'] * lbp_score +
        weights['dct'] * dct_score +
        weights['edge'] * edge_score +
        weights['color'] * color_score
    )
    
    return {
        'variance_score': variance_score,
        'lbp_score': lbp_score,
        'dct_score': dct_score,
        'edge_score': edge_score,
        'color_score': color_score,
        'local_variance': local_var,
        'lbp_entropy': lbp_entropy,
        'dct_ratio': dct_ratio,
        'edge_density': edge_density,
        'color_std': color_std,
        'combined_score': combined,
        'is_likely_obstruction': combined > 0.55
    }


# =============================================================================
# CARD DETECTION AND WARPING
# =============================================================================

def find_card_quad_edges(img: np.ndarray) -> Optional[np.ndarray]:
    """Find card quadrilateral using edge detection and contour analysis."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Canny thresholds based on image median
    v = float(np.median(gray))
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    
    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    edges = cv2.erode(edges, np.ones((5, 5), np.uint8), iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    H, W = img.shape[:2]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Try to find a 4-sided polygon
    for c in contours[:15]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > 0.15 * H * W:
                return approx.reshape(4, 2)
    
    # Fallback: use minAreaRect of largest contour
    c = contours[0]
    if cv2.contourArea(c) > 0.20 * H * W:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        return box
    
    return None


def find_card_quad_brightness(img: np.ndarray) -> Optional[np.ndarray]:
    """Find card quadrilateral using brightness/saturation thresholding."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    
    v_thr = float(np.percentile(V, 55))
    s_thr = float(np.percentile(S, 85))
    
    # Card is bright and relatively unsaturated
    mask = ((V > v_thr) & (S < max(180.0, s_thr))).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 9)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8), iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return box


def warp_card(img: np.ndarray, out_w: int = WARP_W, out_h: int = WARP_H) -> Optional[WarpResult]:
    """Detect card and warp to canonical rectangle."""
    H, W = img.shape[:2]
    
    # Try edge-based detection first
    quad = find_card_quad_edges(img)
    if quad is None or cv2.contourArea(np.array(quad, dtype=np.float32)) < 0.10 * H * W:
        quad = find_card_quad_brightness(img)
    
    if quad is None:
        return None
    
    rect = order_points(quad)
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h))
    
    return WarpResult(warped=warped, M=M, quad=rect)


def crop_roi(warped: np.ndarray, roi_norm: Tuple[float, float, float, float]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop a normalized ROI from the warped card image."""
    x0, y0, x1, y1 = roi_norm
    h, w = warped.shape[:2]
    
    X0 = int(x0 * w)
    X1 = int(x1 * w)
    Y0 = int(y0 * h)
    Y1 = int(y1 * h)
    
    # Clamp to image bounds
    X0 = max(0, min(w - 1, X0))
    X1 = max(1, min(w, X1))
    Y0 = max(0, min(h - 1, Y0))
    Y1 = max(1, min(h, Y1))
    
    return warped[Y0:Y1, X0:X1].copy(), (X0, Y0, X1, Y1)


# =============================================================================
# SIDE DETECTION
# =============================================================================

class FaceDetector:
    """Face detector using Haar cascades with caching."""
    
    _cascade: Optional[cv2.CascadeClassifier] = None
    _profile_cascade: Optional[cv2.CascadeClassifier] = None
    
    @classmethod
    def get_cascade(cls) -> cv2.CascadeClassifier:
        if cls._cascade is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cls._cascade = cv2.CascadeClassifier(cascade_path)
        return cls._cascade
    
    @classmethod
    def get_profile_cascade(cls) -> cv2.CascadeClassifier:
        if cls._profile_cascade is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
            cls._profile_cascade = cv2.CascadeClassifier(cascade_path)
        return cls._profile_cascade
    
    @classmethod
    def detect_face(cls, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in image, returns bbox (x, y, w, h) or None."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Try frontal face first
        cascade = cls.get_cascade()
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            # Return largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            return tuple(faces[0])
        
        # Try profile face as fallback
        profile_cascade = cls.get_profile_cascade()
        faces = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            return tuple(faces[0])
        
        return None


def compute_qr_signature_score(bgr: np.ndarray) -> float:
    """
    Compute a score indicating presence of QR code texture.
    QR codes have high edge density, checkerboard-like patterns, and high local variance.
    Returns score in [0, 1], higher = more likely QR.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1. Edge density - QR codes have lots of edges
    edge_density = compute_edge_density(bgr)
    edge_score = min(1.0, edge_density / 0.25)  # Normalize: 0.25 edge density = score 1.0
    
    # 2. Black/white ratio - QR codes have significant black areas
    black_ratio = compute_black_ratio(bgr, threshold=100)
    # QR typically has 30-50% black pixels
    black_score = 1.0 - abs(black_ratio - 0.40) * 2.5
    black_score = max(0, min(1, black_score))
    
    # 3. Local variance - QR codes have high local variance (texture)
    variance = compute_local_variance(gray, kernel_size=11)
    mean_var = float(np.mean(variance))
    var_score = min(1.0, mean_var / 2000)  # Normalize
    
    # 4. Histogram bimodality - QR codes have bimodal histogram (black & white)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    # Check if peaks exist near 0 and 255
    dark_peak = hist[:80].max()
    light_peak = hist[180:].max()
    bimodal_score = min(1.0, (dark_peak + light_peak) * 10)
    
    # Combined score
    score = 0.35 * edge_score + 0.25 * black_score + 0.25 * var_score + 0.15 * bimodal_score
    return float(score)


def detect_mrz_lines(warped: np.ndarray) -> int:
    """Detect number of MRZ-like text lines in the bottom portion of the card."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Focus on bottom ~30% where MRZ would be
    region = gray[int(h * 0.68):int(h * 0.98), :]
    region = cv2.GaussianBlur(region, (3, 3), 0)
    
    # Adaptive threshold to find text
    thr = cv2.adaptiveThreshold(
        region, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )
    
    # Horizontal morphology to merge characters into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 3))
    merged = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    merged = cv2.morphologyEx(
        merged,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)),
        iterations=1
    )
    
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lines = 0
    for c in contours:
        x, y, wc, hc = cv2.boundingRect(c)
        # MRZ lines are wide and thin
        if wc > 0.55 * w and hc < 0.35 * region.shape[0]:
            lines += 1
    
    return lines


def detect_text_layout(warped: np.ndarray) -> Dict[str, float]:
    """
    Detect text layout patterns to help distinguish front from back.
    Front cards have structured text (NOMBRE, DOMICILIO, etc.)
    Back cards have QR codes and MRZ.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Check for "NOMBRE" text region (front indicator)
    # This appears in the upper-middle area on front cards
    nombre_region = gray[int(h*0.20):int(h*0.35), int(w*0.28):int(w*0.55)]
    nombre_edge = compute_edge_density(cv2.cvtColor(
        cv2.merge([nombre_region, nombre_region, nombre_region]), cv2.COLOR_BGR2GRAY
    ).reshape(nombre_region.shape[0], nombre_region.shape[1], 1).repeat(3, axis=2))
    
    # Check for barcode at top (back indicator)
    barcode_region = gray[int(h*0.02):int(h*0.12), int(w*0.05):int(w*0.25)]
    barcode_edge = compute_edge_density(cv2.cvtColor(
        cv2.merge([barcode_region, barcode_region, barcode_region]), cv2.COLOR_BGR2GRAY
    ).reshape(barcode_region.shape[0], barcode_region.shape[1], 1).repeat(3, axis=2))
    
    # Check hologram region (front indicator - has faint image)
    hologram_region = gray[int(h*0.25):int(h*0.65), int(w*0.74):int(w*0.96)]
    hologram_std = float(np.std(hologram_region))
    
    # Check for Mexican coat of arms in top-left (front indicator)
    coat_region = gray[int(h*0.05):int(h*0.22), int(w*0.05):int(w*0.20)]
    coat_edge = compute_edge_density(cv2.cvtColor(
        cv2.merge([coat_region, coat_region, coat_region]), cv2.COLOR_BGR2GRAY
    ).reshape(coat_region.shape[0], coat_region.shape[1], 1).repeat(3, axis=2))
    
    return {
        "nombre_edge": nombre_edge if nombre_region.size > 0 else 0,
        "barcode_edge": barcode_edge if barcode_region.size > 0 else 0,
        "hologram_std": hologram_std,
        "coat_edge": coat_edge if coat_region.size > 0 else 0,
    }


def classify_side(warped: np.ndarray) -> SideDetectionResult:
    """
    Classify whether the card shows front or back side.
    
    Uses multiple signals:
    1. Face detection in main_photo region (strong front indicator)
    2. QR signature in qr_left/qr_mid regions (strong back indicator)
    3. MRZ line detection (back indicator)
    4. Text layout analysis (front has NOMBRE, coat of arms; back has barcode)
    5. Hologram region analysis (front has faint secondary photo)
    """
    h, w = warped.shape[:2]
    
    # Extract main_photo region for face detection
    photo_roi = FRONT_ROIS["main_photo"]
    photo_crop, _ = crop_roi(warped, photo_roi)
    
    # Detect face
    face_bbox = FaceDetector.detect_face(photo_crop)
    face_detected = face_bbox is not None
    
    # Extract QR regions for signature detection
    qr_left_crop, _ = crop_roi(warped, BACK_ROIS["qr_left"])
    qr_mid_crop, _ = crop_roi(warped, BACK_ROIS["qr_mid"])
    
    qr_score_left = compute_qr_signature_score(qr_left_crop)
    qr_score_mid = compute_qr_signature_score(qr_mid_crop)
    qr_signature_score = max(qr_score_left, qr_score_mid)
    
    # Detect MRZ lines
    mrz_lines = detect_mrz_lines(warped)
    
    # Analyze text layout
    layout = detect_text_layout(warped)
    
    # Decision logic with confidence
    front_score = 0.0
    back_score = 0.0
    
    # Face is a very strong front indicator
    if face_detected:
        front_score += 0.7
    
    # QR signature is a strong back indicator (only if score is high)
    if qr_signature_score > 0.6:
        back_score += 0.5 * qr_signature_score
    elif qr_signature_score > 0.5:
        back_score += 0.3 * qr_signature_score
    
    # MRZ lines are a moderate back indicator
    if mrz_lines >= 2:
        back_score += 0.3
    elif mrz_lines == 1:
        back_score += 0.15
    
    # Text layout analysis
    # High barcode edge density = back
    if layout["barcode_edge"] > 0.15:
        back_score += 0.25
    
    # Coat of arms presence = front (moderate edge in that region)
    if 0.05 < layout["coat_edge"] < 0.20:
        front_score += 0.15
    
    # Hologram region has moderate variance on front (faint photo)
    if layout["hologram_std"] > 20:
        front_score += 0.1
    
    # If neither face nor strong QR detected, use additional heuristics
    if not face_detected and qr_signature_score < 0.5:
        # Check if photo region has photo-like texture (skin tones, edges)
        photo_edge = compute_edge_density(photo_crop)
        
        # Photos typically have moderate edge density (0.05-0.15)
        if 0.04 < photo_edge < 0.18:
            front_score += 0.2
        
        # Check for bright flat blob in photo area (potential obstruction covering face)
        photo_blob = detect_bright_flat_blob(photo_crop, v_perc=70, mag_perc=50, min_area_frac=0.10)
        if photo_blob is not None and photo_blob.rectangularity > 0.5:
            # There's likely an obstruction - could be covering a face
            # Don't penalize front score, but don't add to back either
            front_score += 0.15
        
        # QR regions on front cards have low edge density (text areas)
        qr_left_edge = compute_edge_density(qr_left_crop)
        qr_mid_edge = compute_edge_density(qr_mid_crop)
        
        # True QR codes have edge density > 0.15
        if qr_left_edge > 0.18 and qr_mid_edge > 0.15:
            back_score += 0.25
        elif qr_left_edge < 0.12 and qr_mid_edge < 0.12:
            # Low edge in QR regions = likely front (text areas)
            front_score += 0.15
    
    # Determine side and confidence
    if front_score > back_score:
        side = "front"
        confidence = min(0.95, front_score / max(0.01, front_score + back_score))
    else:
        side = "back"
        confidence = min(0.95, back_score / max(0.01, front_score + back_score))
    
    # Minimum confidence threshold
    confidence = max(0.5, confidence)
    
    return SideDetectionResult(
        side=side,
        confidence=confidence,
        face_detected=face_detected,
        face_bbox=face_bbox,
        qr_signature_score=qr_signature_score,
        mrz_lines=mrz_lines
    )


# =============================================================================
# OBSTRUCTION DETECTION
# =============================================================================

def detect_bright_flat_blob(
    bgr: np.ndarray,
    v_perc: int = 70,
    mag_perc: int = 45,
    min_area_frac: float = 0.07,
) -> Optional[BlobInfo]:
    """
    Detect bright, flat (low gradient) regions that could be obstructions.
    
    Paper and sticky notes appear as:
    - Bright (high V in HSV)
    - Flat (low gradient magnitude)
    - Uniform color (low color std dev)
    - Sharp edges
    
    Glare appears as:
    - Bright
    - Gradient edges (not sharp)
    - Less uniform (color varies)
    """
    h, w = bgr.shape[:2]
    if h < 10 or w < 10:
        return None
    
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    
    v_thr = float(np.percentile(V, v_perc))
    
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (9, 9), 0)
    m_thr = float(np.percentile(mag, mag_perc))
    
    # Create mask for bright, flat regions
    mask = ((V > v_thr) & (mag < m_thr)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8), iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    
    if area < min_area_frac * h * w:
        return None
    
    x, y, wc, hc = cv2.boundingRect(c)
    rect_area = float(wc * hc)
    rectangularity = area / rect_area if rect_area > 0 else 0.0
    
    # Analyze blob properties
    blob_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(blob_mask, [c], -1, 255, -1)
    
    # Color uniformity (low std = uniform = likely paper)
    blob_pixels = bgr[blob_mask > 0]
    if len(blob_pixels) > 0:
        color_std = float(np.std(blob_pixels))
    else:
        color_std = 999.0
    
    # Mean saturation (high = colored sticky note, low = white paper or glare)
    mean_sat = float(np.mean(S[blob_mask > 0])) if np.any(blob_mask > 0) else 0.0
    
    # Edge sharpness - paper has sharp edges, glare has gradient edges
    # Compute gradient at blob boundary
    boundary = cv2.dilate(blob_mask, np.ones((5, 5), np.uint8)) - cv2.erode(blob_mask, np.ones((5, 5), np.uint8))
    if np.any(boundary > 0):
        boundary_grad = mag[boundary > 0]
        edge_sharpness = float(np.mean(boundary_grad))
    else:
        edge_sharpness = 0.0
    
    return BlobInfo(
        bbox=(int(x), int(y), int(wc), int(hc)),
        area_frac=float(area / (h * w)),
        rectangularity=float(rectangularity),
        color_std=color_std,
        mean_saturation=mean_sat,
        edge_sharpness=edge_sharpness,
        v_thr=v_thr,
        m_thr=m_thr,
    )


def detect_colored_blob(
    bgr: np.ndarray,
    min_saturation: int = 60,
    min_area_frac: float = 0.05,
) -> Optional[BlobInfo]:
    """
    Detect highly saturated (colored) blobs like pink/yellow sticky notes.
    
    Colored sticky notes have:
    - High saturation (S > 60)
    - Moderate to high value (not dark)
    - Relatively uniform hue
    - Hue NOT in skin tone range (exclude ~0-25 and ~170-180) UNLESS very high sat
    
    This catches sticky notes that the brightness-based detection might miss.
    """
    h, w = bgr.shape[:2]
    if h < 10 or w < 10:
        return None
    
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    
    # Create mask for highly saturated regions
    # EXCLUDE skin tones: H in [0, 25] or [165, 180] with S in [30, 80]
    # Note: Pink sticky notes have S > 80 even with pink hue, so don't exclude those
    is_high_sat = (S > min_saturation) & (V > 100)
    is_skin_tone = ((H < 25) | (H > 165)) & (S > 30) & (S < 80)  # Changed from S < 160 to S < 80
    
    sat_mask = (is_high_sat & ~is_skin_tone).astype(np.uint8) * 255
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
    
    contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    
    if area < min_area_frac * h * w:
        return None
    
    x, y, wc, hc = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    rectangularity = area / hull_area if hull_area > 0 else 0
    
    # Analyze blob properties
    blob_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(blob_mask, [c], -1, 255, -1)
    
    blob_pixels = bgr[blob_mask > 0]
    color_std = float(np.std(blob_pixels)) if len(blob_pixels) > 0 else 999.0
    mean_sat = float(np.mean(S[blob_mask > 0])) if np.any(blob_mask > 0) else 0.0
    mean_hue = float(np.mean(H[blob_mask > 0])) if np.any(blob_mask > 0) else 0.0
    
    # Additional filter: Check if this is actually a non-skin colored blob
    # Pink sticky notes typically have H around 0-10 or 165-180 with VERY high saturation
    # Yellow sticky notes have H around 25-35
    # Green/Blue/etc have H in mid ranges
    is_sticky_color = (
        (mean_hue > 25 and mean_hue < 165) or  # Non-skin hue range (yellow, green, blue, etc)
        (mean_sat > 70)  # Very high saturation - even pink is a sticky, not skin (lowered from 80)
    )
    
    if not is_sticky_color:
        return None
    
    # Compute edge sharpness
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    
    boundary = cv2.dilate(blob_mask, np.ones((5, 5), np.uint8)) - cv2.erode(blob_mask, np.ones((5, 5), np.uint8))
    edge_sharpness = float(np.mean(mag[boundary > 0])) if np.any(boundary > 0) else 0.0
    
    return BlobInfo(
        bbox=(int(x), int(y), int(wc), int(hc)),
        area_frac=float(area / (h * w)),
        rectangularity=float(rectangularity),
        color_std=color_std,
        mean_saturation=mean_sat,
        edge_sharpness=edge_sharpness,
        v_thr=0.0,
        m_thr=0.0,
    )


def detect_partial_coverage(bgr: np.ndarray) -> Tuple[bool, float, str]:
    """
    Detect if a region is partially covered by analyzing edge density variance
    across quadrants. If one part has high edge density and another has very low,
    it indicates partial obstruction.
    
    V5: DISABLED - this was causing too many false positives.
    
    Returns: (is_partially_covered, coverage_fraction, description)
    """
    # V5: Disabled due to high false positive rate
    return False, 0.0, ""


def detect_white_blob_in_qr(bgr: np.ndarray, min_area_frac: float = 0.12) -> Optional[BlobInfo]:
    """
    Detect white/light uniform regions in QR areas that could be paper obstructions.
    
    This specifically targets white paper that:
    - Has high brightness (Value > 160)
    - Has low saturation (< 35, i.e., gray/white)
    - Has low local variance (uniform)
    - Is a significant area (>12%)
    - Has very low color std (<8)
    
    V5: Made MUCH stricter to reduce false positives.
    
    Returns blob info if white obstruction is found.
    """
    h, w = bgr.shape[:2]
    if h < 20 or w < 20:
        return None
    
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # White/neutral regions: high V, low S (stricter: S < 35 instead of 40)
    v_threshold = np.percentile(V, 70)  # Top 30% brightness (stricter)
    white_mask = ((V > v_threshold) & (S < 35)).astype(np.uint8) * 255
    
    # Compute local variance to find uniform regions (stricter: var < 150)
    variance = compute_local_variance(gray, kernel_size=11)
    low_var_mask = (variance < 150).astype(np.uint8) * 255
    
    # Combine: bright, unsaturated, low variance
    combined_mask = cv2.bitwise_and(white_mask, low_var_mask)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=1)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    
    if area < min_area_frac * h * w:
        return None
    
    x, y, wc, hc = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    rectangularity = area / hull_area if hull_area > 0 else 0
    
    # Check that it's actually a uniform blob
    blob_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(blob_mask, [c], -1, 255, -1)
    
    blob_pixels = bgr[blob_mask > 0]
    color_std = float(np.std(blob_pixels)) if len(blob_pixels) > 0 else 999.0
    mean_sat = float(np.mean(S[blob_mask > 0])) if np.any(blob_mask > 0) else 0.0
    
    # V8: Relaxed uniformity check (std < 12, was 8)
    if color_std > 12:
        return None
    
    # V8: Relaxed rectangularity (> 0.30, was 0.35)
    if rectangularity < 0.30:
        return None
    
    # Compute edge sharpness at boundary
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    
    boundary = cv2.dilate(blob_mask, np.ones((5, 5), np.uint8)) - cv2.erode(blob_mask, np.ones((5, 5), np.uint8))
    edge_sharpness = float(np.mean(mag[boundary > 0])) if np.any(boundary > 0) else 0.0
    
    return BlobInfo(
        bbox=(int(x), int(y), int(wc), int(hc)),
        area_frac=float(area / (h * w)),
        rectangularity=float(rectangularity),
        color_std=color_std,
        mean_saturation=mean_sat,
        edge_sharpness=edge_sharpness,
        v_thr=float(v_threshold),
        m_thr=0.0,
    )


def is_obstruction_blob(blob: BlobInfo, thresholds: Dict, roi_name: str = "") -> Tuple[bool, float, str]:
    """
    Determine if a detected blob is actually an obstruction vs glare/artifact.
    
    V7: Enhanced with region-specific thresholds
    - Special handling for QR regions (especially qr_mid which has many FPs)
    - main_photo: requires very low std OR high saturation
    - Design element filter is stricter
    
    Returns: (is_obstruction, confidence, reason)
    """
    reasons = []
    confidence = 0.0
    
    min_rect = thresholds.get("blob_min_rect", 0.55)
    color_std_max = thresholds.get("color_std_max", 35)
    
    # V7: Special handling for QR regions - require very low std
    # QR codes have high contrast texture, so obstructions stand out as uniform
    # qr_mid has frequent FPs from reflections, so extra strict
    if roi_name in {"qr_left", "qr_mid", "qr_small"}:
        if roi_name == "qr_mid":
            # V7: qr_mid needs VERY strict criteria
            # FP analysis: FPs have std=17-22; real obstructions have std<10
            if blob.color_std > 10:
                return False, 0.0, f"qr_mid: std too high ({blob.color_std:.1f})"
        elif roi_name == "qr_small":
            # V7: qr_small can be slightly more lenient
            # Image 30 has std=18.8 which should be caught
            if blob.color_std > 20:
                return False, 0.0, f"qr_small: std too high ({blob.color_std:.1f})"
        else:  # qr_left
            if blob.color_std > 15:
                return False, 0.0, f"qr_left: std too high ({blob.color_std:.1f})"
    
    # V8: Relaxed design element exclusion
    # Now requires std < 15 OR saturation > 40 OR rectangularity > 0.60 (was std<12, rect>0.70)
    if blob.color_std > 15 and blob.mean_saturation < 40 and blob.rectangularity < 0.60:
        return False, 0.0, "design element (high std, low sat, low rect)"
    
    # V5: Special handling for main_photo region
    # V8: Relaxed rectangularity requirement (0.30 instead of 0.40)
    if roi_name == "main_photo":
        if blob.mean_saturation > 50:
            pass  # Colored obstruction - allow to proceed
        elif blob.color_std < 5 and blob.rectangularity > 0.30 and blob.area_frac > 0.06:
            pass  # Very uniform AND somewhat rectangular - likely paper
        # V8: Also detect very uniform blobs regardless of shape
        elif blob.color_std < 4 and blob.area_frac > 0.08:
            pass  # Extremely uniform - almost certainly an obstruction
        else:
            return False, 0.0, "main_photo: not colored enough and not uniform+rectangular enough"
    
    # V5: Special handling for hologram_photo
    if roi_name == "hologram_photo":
        if blob.color_std > 15:
            return False, 0.0, "hologram: color_std too high for obstruction"
    
    # V5: Special handling for lower_left (signature area)
    # V8: Use area as main discriminator - real obstructions tend to be larger
    if roi_name == "lower_left":
        if blob.mean_saturation > 50:
            return True, 0.65, f"lower_left colored obstruction (sat={blob.mean_saturation:.1f})"
        # V8: Large blobs (>18% area) with moderate uniformity - very likely obstruction
        elif blob.area_frac > 0.18 and blob.color_std < 10 and blob.rectangularity > 0.35:
            return True, 0.55, f"lower_left large uniform (area={blob.area_frac:.1%}, std={blob.color_std:.1f})"
        # V8: Smaller but extremely uniform blobs (std<4)
        elif blob.color_std < 4 and blob.area_frac > 0.10 and blob.rectangularity > 0.40:
            return True, 0.50, f"lower_left very uniform (std={blob.color_std:.1f}, area={blob.area_frac:.1%})"
        else:
            return False, 0.0, "lower_left: not meeting criteria"
    
    # For VERY uniform blobs (std < 6), allow lower rectangularity
    # V8: Even lower threshold for extremely uniform blobs
    effective_min_rect = min_rect
    if blob.color_std < 4:
        effective_min_rect = min(min_rect, 0.25)  # Very low for extremely uniform
    elif blob.color_std < 6:
        effective_min_rect = min(min_rect, 0.35)  # Reduced from 0.38
    
    # Rectangularity check - paper/sticky notes are rectangular
    if blob.rectangularity >= effective_min_rect:
        rect_conf = min(1.0, (blob.rectangularity - effective_min_rect) / 0.3 + 0.5)
        confidence += 0.35 * rect_conf
        reasons.append(f"rectangular shape ({blob.rectangularity:.2f})")
    else:
        confidence -= 0.25  # Stronger penalty
    
    # Color uniformity check - paper has uniform color
    if blob.color_std < color_std_max:
        unif_conf = 1.0 - (blob.color_std / color_std_max)
        confidence += 0.25 * unif_conf
        reasons.append(f"uniform color (std={blob.color_std:.1f})")
    
    # Area check - reasonable obstruction size
    if blob.area_frac > 0.10:  # Raised from 0.08
        area_conf = min(1.0, blob.area_frac / 0.3)
        confidence += 0.2 * area_conf
        reasons.append(f"significant area ({blob.area_frac:.2%})")
    
    # Edge sharpness - paper has sharper edges than glare
    # But VERY uniform blobs (std < 5) are likely paper even with soft edges
    if blob.edge_sharpness > 35:
        sharp_conf = min(1.0, blob.edge_sharpness / 100)
        confidence += 0.12 * sharp_conf
        reasons.append(f"sharp edges ({blob.edge_sharpness:.1f})")
    elif blob.edge_sharpness < 15 and blob.color_std >= 5:
        # Only penalize soft edges if NOT very uniform
        confidence -= 0.15
    
    # Saturation check - colored sticky notes have higher saturation
    if blob.mean_saturation > 50:
        confidence += 0.20  # Increased bonus for colored
        reasons.append(f"colored ({blob.mean_saturation:.0f} sat)")
    
    # V8: Confidence threshold of 0.38 (reduced from 0.42 to catch more real obstructions)
    is_obstruction = confidence > 0.38
    reason = "; ".join(reasons) if reasons else "no obstruction signals"
    
    return is_obstruction, confidence, reason


def detect_roi_obstruction(
    crop: np.ndarray,
    side: str,
    roi_name: str,
    face_detected: bool = False,
) -> ObstructionResult:
    """
    Detect if a ROI is obstructed.
    
    V7 uses multiple signals with advanced analysis:
    1. Bright flat blob detection
    2. High-saturation colored blob detection (pink/yellow stickies)
    3. Edge density analysis
    4. NEW: LBP texture entropy (low entropy = uniform = obstruction)
    5. NEW: DCT frequency analysis (high DC ratio = uniform = obstruction)
    6. NEW: Glare vs paper classification using gradient profiles
    7. For QR regions: black ratio and texture analysis
    
    Args:
        crop: Cropped ROI image
        side: 'front' or 'back'
        roi_name: Name of the ROI
        face_detected: If True and roi_name is 'main_photo', skip obstruction detection
                       (visible face means no obstruction)
    """
    # V5: If face was detected and this is main_photo, don't flag as obstructed
    if roi_name == "main_photo" and face_detected:
        return ObstructionResult(
            obstructed=False,
            confidence=0.0,
            reason="face visible (not obstructed)",
            edge_density=compute_edge_density(crop),
            blob=None,
            black_ratio=None,
            metrics={"face_detected": True},
        )
    
    thresholds = THRESHOLDS.get(side, {}).get(roi_name, {})
    
    # Get thresholds with defaults
    min_edge = thresholds.get("min_edge_density", 0.04)
    blob_v = thresholds.get("blob_v_perc", 70)
    blob_mag = thresholds.get("blob_mag_perc", 50)
    blob_area = thresholds.get("blob_min_area", 0.08)
    min_black = thresholds.get("min_black_ratio", None)
    
    metrics: Dict[str, Any] = {}
    
    # Compute edge density
    edge_density = compute_edge_density(crop)
    metrics["edge_density"] = edge_density
    
    # V7 NEW: Compute advanced texture/frequency metrics
    adv_scores = compute_obstruction_score(crop)
    metrics["lbp_entropy"] = adv_scores["lbp_entropy"]
    metrics["dct_ratio"] = adv_scores["dct_ratio"]
    metrics["combined_score"] = adv_scores["combined_score"]
    
    # Detect bright flat blob
    blob = detect_bright_flat_blob(
        crop,
        v_perc=blob_v,
        mag_perc=blob_mag,
        min_area_frac=blob_area
    )
    
    # Also detect colored blobs (pink/yellow sticky notes)
    colored_blob = detect_colored_blob(crop, min_saturation=60, min_area_frac=0.05)
    
    # Check for obstruction
    obstructed = False
    confidence = 0.0
    reasons = []
    
    # 1. Colored blob detection (pink/yellow stickies) - check this FIRST
    if colored_blob is not None and colored_blob.mean_saturation > 70 and roi_name != "lower_left":
        is_design_color = False
        if side == "front":
            h, w = crop.shape[:2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            H = hsv[:, :, 0].astype(np.float32)
            S = hsv[:, :, 1].astype(np.float32)
            high_sat_mask = S > 70
            if np.sum(high_sat_mask) > 0:
                mean_hue = float(np.mean(H[high_sat_mask]))
                if 10 < mean_hue < 40:
                    is_design_color = True
        
        if not is_design_color and colored_blob.area_frac > 0.05 and colored_blob.rectangularity > 0.30:
            obstructed = True
            confidence = max(confidence, 0.75)
            reasons.append(f"colored obstruction (sat={colored_blob.mean_saturation:.0f}, area={colored_blob.area_frac:.1%})")
    
    # 2. Bright flat blob detection (white paper) with V7 glare classification
    if blob is not None and not obstructed:
        # V7 NEW: Check if this is glare or paper using gradient profile
        # Get contour for glare classification
        h, w = crop.shape[:2]
        x, y, wc, hc = blob.bbox
        contour = np.array([[x, y], [x+wc, y], [x+wc, y+hc], [x, y+hc]])
        
        glare_class = classify_glare_vs_paper(crop, contour)
        metrics["glare_classification"] = glare_class
        
        # If classified as glare with high confidence, skip
        if glare_class["is_glare"] and glare_class["confidence"] > 0.6:
            # It's glare, not an obstruction - skip blob detection
            pass
        else:
            is_obs, blob_conf, blob_reason = is_obstruction_blob(blob, thresholds, roi_name)
            if is_obs:
                # V7: Boost confidence if LBP/DCT also indicate obstruction
                if adv_scores["lbp_entropy"] < 3.0:
                    blob_conf = min(1.0, blob_conf + 0.1)
                if adv_scores["dct_ratio"] > 0.5:
                    blob_conf = min(1.0, blob_conf + 0.1)
                
                obstructed = True
                confidence = max(confidence, blob_conf)
                reasons.append(f"blob: {blob_reason}")
    
    # 3. QR-specific detection
    if side == "back" and roi_name in {"qr_left", "qr_mid", "qr_small"}:
        if not obstructed:
            white_blob = detect_white_blob_in_qr(crop, min_area_frac=0.12)
            if white_blob is not None and white_blob.area_frac > 0.12 and white_blob.color_std < 10:
                obstructed = True
                confidence = max(confidence, 0.65)
                reasons.append(f"white obstruction on QR (area={white_blob.area_frac:.1%}, std={white_blob.color_std:.1f})")
                if blob is None or white_blob.area_frac > blob.area_frac:
                    blob = white_blob
        
        # Edge density check for full coverage
        if edge_density < min_edge * 0.8 and not obstructed:
            black_ratio = compute_black_ratio(crop)
            metrics["black_ratio"] = black_ratio
            
            expected_black = min_black if min_black else 0.15
            if black_ratio < expected_black * 0.4:
                obstructed = True
                confidence = max(confidence, 0.55)
                reasons.append(f"missing QR texture (edge={edge_density:.3f}, black={black_ratio:.3f})")
        
        # V7: Stricter uniform blob check for QR
        # qr_mid needs stricter thresholds due to frequent FPs from reflections
        if blob is not None and not obstructed:
            if roi_name == "qr_mid":
                # V7: Much stricter for qr_mid - require BOTH low std AND high rect
                # FP analysis showed: FPs have std=17-22, TNs have similar std but lower rect
                if blob.area_frac > 0.15 and blob.color_std < 12 and blob.rectangularity > 0.65:
                    obstructed = True
                    confidence = max(confidence, 0.55)
                    reasons.append(f"uniform blob on QR (area={blob.area_frac:.1%}, std={blob.color_std:.1f}, rect={blob.rectangularity:.2f})")
            else:
                # qr_left and qr_small can use slightly relaxed thresholds
                if blob.area_frac > 0.12 and blob.color_std < 12 and blob.rectangularity > 0.45:
                    obstructed = True
                    confidence = max(confidence, 0.55)
                    reasons.append(f"uniform blob on QR (area={blob.area_frac:.1%}, std={blob.color_std:.1f})")
    
    elif side == "back" and roi_name == "mrz":
        # V8: Enhanced MRZ detection - check for white blobs covering MRZ
        # First check for white blob obstruction (like QR regions)
        if not obstructed:
            white_blob = detect_white_blob_in_qr(crop, min_area_frac=0.08)  # Lower threshold for MRZ
            if white_blob is not None and white_blob.area_frac > 0.08 and white_blob.color_std < 15:
                obstructed = True
                confidence = max(confidence, 0.60)
                reasons.append(f"white obstruction on MRZ (area={white_blob.area_frac:.1%}, std={white_blob.color_std:.1f})")
                if blob is None or white_blob.area_frac > blob.area_frac:
                    blob = white_blob
        
        # V8: Check bright flat blob on MRZ
        if blob is not None and not obstructed:
            if blob.area_frac > 0.08 and blob.color_std < 15 and blob.rectangularity > 0.35:
                obstructed = True
                confidence = max(confidence, 0.55)
                reasons.append(f"uniform blob on MRZ (area={blob.area_frac:.1%}, std={blob.color_std:.1f}, rect={blob.rectangularity:.2f})")
        
        # MRZ detection - also use LBP entropy
        if edge_density < min_edge * 0.8 and not obstructed:
            # V7: Also check LBP entropy - MRZ text should have high entropy
            if adv_scores["lbp_entropy"] < 2.5:
                obstructed = True
                confidence = max(confidence, 0.55)
                reasons.append(f"missing MRZ (edge={edge_density:.3f}, lbp={adv_scores['lbp_entropy']:.2f})")
    
    # 4. V7 NEW: Multi-signal fusion for ambiguous cases
    # If traditional methods didn't detect but combined score is very high, flag it
    if not obstructed and adv_scores["combined_score"] > 0.70:
        # Additional check: require at least 2 strong signals
        strong_signals = 0
        if adv_scores["lbp_entropy"] < 2.5:
            strong_signals += 1
        if adv_scores["dct_ratio"] > 0.55:
            strong_signals += 1
        if edge_density < 0.03:
            strong_signals += 1
        if adv_scores["color_std"] < 15:
            strong_signals += 1
        
        if strong_signals >= 2:
            obstructed = True
            confidence = max(confidence, 0.50)
            reasons.append(f"multi-signal detection (score={adv_scores['combined_score']:.2f}, lbp={adv_scores['lbp_entropy']:.2f}, dct={adv_scores['dct_ratio']:.2f})")
    
    # 5. V7 NEW: Text region detection using LBP/DCT
    # Text areas should have high texture entropy
    if side == "front" and roi_name in {"text_center", "text_bottom"} and not obstructed:
        # Text should have high LBP entropy (>3.5) and low DCT ratio (<0.4)
        if adv_scores["lbp_entropy"] < 2.8 and adv_scores["dct_ratio"] > 0.45:
            obstructed = True
            confidence = max(confidence, 0.50)
            reasons.append(f"uniform region in text area (lbp={adv_scores['lbp_entropy']:.2f}, dct={adv_scores['dct_ratio']:.2f})")
    
    # Build reason string
    if obstructed:
        reason = "; ".join(reasons)
    else:
        reason = "no obstruction detected"
    
    # Use the most informative blob for output
    output_blob = colored_blob if (colored_blob and colored_blob.mean_saturation > 70) else blob
    
    # Convert blob to serializable format
    blob_info = None
    if output_blob is not None:
        blob_info = BlobInfo(
            bbox=output_blob.bbox,
            area_frac=output_blob.area_frac,
            rectangularity=output_blob.rectangularity,
            color_std=output_blob.color_std,
            mean_saturation=output_blob.mean_saturation,
            edge_sharpness=output_blob.edge_sharpness,
            v_thr=output_blob.v_thr,
            m_thr=output_blob.m_thr,
        )
    
    return ObstructionResult(
        obstructed=obstructed,
        confidence=confidence,
        reason=reason,
        edge_density=edge_density,
        blob=blob_info,
        black_ratio=metrics.get("black_ratio"),
        metrics=metrics,
    )


def detect_all_obstructions(warped: np.ndarray, side: str, face_detected: bool = False) -> Dict[str, Dict]:
    """Detect obstructions in all ROIs for the given side.
    
    Args:
        warped: Warped card image
        side: 'front' or 'back'
        face_detected: Whether face was detected during side classification.
                       If True, main_photo region won't be flagged as obstructed
                       (visible face means no obstruction over the face).
    """
    rois = FRONT_ROIS if side == "front" else BACK_ROIS
    results: Dict[str, Dict] = {}
    
    for name, roi in rois.items():
        crop, px = crop_roi(warped, roi)
        result = detect_roi_obstruction(crop, side, name, face_detected=face_detected)
        
        # Convert to dict for JSON serialization
        blob_dict = None
        if result.blob is not None:
            blob_dict = {
                "bbox": result.blob.bbox,
                "area_frac": result.blob.area_frac,
                "rectangularity": result.blob.rectangularity,
                "color_std": result.blob.color_std,
                "mean_saturation": result.blob.mean_saturation,
                "edge_sharpness": result.blob.edge_sharpness,
                "v_thr": result.blob.v_thr,
                "m_thr": result.blob.m_thr,
            }
        
        results[name] = {
            "roi_px": px,
            "obstructed": result.obstructed,
            "confidence": result.confidence,
            "reason": result.reason,
            "detail": {
                "edge_density": result.edge_density,
                "blob": blob_dict,
                "black_ratio": result.black_ratio,
                "metrics": result.metrics,
            }
        }
    
    return results


# =============================================================================
# DEBUG VISUALIZATION
# =============================================================================

def draw_debug(
    warped: np.ndarray,
    side: str,
    detections: Dict[str, Dict],
    side_result: Optional[SideDetectionResult] = None
) -> np.ndarray:
    """Draw debug overlay showing ROIs and detected obstructions."""
    out = warped.copy()
    h, w = out.shape[:2]
    
    for name, info in detections.items():
        x0, y0, x1, y1 = info["roi_px"]
        obstructed = info["obstructed"]
        confidence = info.get("confidence", 0)
        
        # Color: red if obstructed, green if clear
        # Intensity based on confidence
        if obstructed:
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green
        
        # Draw ROI rectangle
        thickness = 2 if not obstructed else 3
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
        
        # Draw ROI label
        label = name
        if obstructed:
            label += f" ({confidence:.0%})"
        
        cv2.putText(
            out, label,
            (x0, max(15, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, color, 1, cv2.LINE_AA
        )
        
        # Draw blob bounding box if detected
        blob = info.get("detail", {}).get("blob")
        if blob is not None and obstructed:
            bx, by, bw, bh = blob["bbox"]
            cv2.rectangle(
                out,
                (x0 + bx, y0 + by),
                (x0 + bx + bw, y0 + by + bh),
                (0, 0, 255), 2
            )
    
    # Draw side label
    side_label = f"side: {side}"
    if side_result:
        side_label += f" ({side_result.confidence:.0%})"
        if side_result.face_detected:
            side_label += " [face]"
        if side_result.qr_signature_score > 0.5:
            side_label += f" [qr:{side_result.qr_signature_score:.2f}]"
    
    cv2.putText(
        out, side_label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 0), 2, cv2.LINE_AA
    )
    
    # Draw face bbox if detected (on front)
    if side_result and side_result.face_detected and side_result.face_bbox:
        photo_roi = FRONT_ROIS.get("main_photo")
        if photo_roi:
            px0 = int(photo_roi[0] * w)
            py0 = int(photo_roi[1] * h)
            fx, fy, fw, fh = side_result.face_bbox
            cv2.rectangle(
                out,
                (px0 + fx, py0 + fy),
                (px0 + fx + fw, py0 + fy + fh),
                (255, 0, 255), 2
            )
    
    return out


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def iter_images(input_path: Path) -> List[Path]:
    """Iterate over image files in path (file or directory)."""
    if input_path.is_file():
        return [input_path]
    
    paths: List[Path] = []
    for p in sorted(input_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    
    return paths


def process_one(
    path: Path,
    debug_dir: Optional[Path] = None,
    force_side: Optional[str] = None,
    log_metrics: bool = False,
) -> Dict:
    """Process a single image."""
    img = cv2.imread(str(path))
    if img is None:
        return {"path": str(path), "ok": False, "error": "could not read image"}
    
    # Warp card to canonical size
    wr = warp_card(img)
    if wr is None:
        return {"path": str(path), "ok": False, "error": "could not detect card quad"}
    
    warped = wr.warped
    
    # Classify side
    if force_side:
        side = force_side
        side_result = None
        face_detected = False
    else:
        side_result = classify_side(warped)
        side = side_result.side
        face_detected = side_result.face_detected
    
    # Detect obstructions (pass face_detected to avoid FPs on visible faces)
    detections = detect_all_obstructions(warped, side, face_detected=face_detected)
    any_obstructed = any(v["obstructed"] for v in detections.values())
    
    # Log metrics for threshold tuning
    if log_metrics:
        logger.info(f"Image: {path.name}")
        logger.info(f"  Side: {side}" + (f" (conf={side_result.confidence:.2f})" if side_result else ""))
        for name, info in detections.items():
            detail = info.get("detail", {})
            logger.info(f"  {name}: obs={info['obstructed']}, edge={detail.get('edge_density', 0):.3f}")
            if detail.get("blob"):
                b = detail["blob"]
                logger.info(f"    blob: rect={b['rectangularity']:.2f}, "
                           f"area={b['area_frac']:.2%}, color_std={b['color_std']:.1f}")
    
    # Save debug image
    debug_path = None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        dbg = draw_debug(warped, side, detections, side_result)
        debug_path = debug_dir / (path.stem + "__debug.png")
        cv2.imwrite(str(debug_path), dbg)
    
    # Build result dict
    result = {
        "path": str(path),
        "ok": True,
        "side": side,
        "obstructed": bool(any_obstructed),
        "regions": detections,
        "debug_image": str(debug_path) if debug_path else None,
    }
    
    # Add side detection details if available
    if side_result:
        result["side_detection"] = {
            "confidence": side_result.confidence,
            "face_detected": side_result.face_detected,
            "qr_signature_score": side_result.qr_signature_score,
            "mrz_lines": side_result.mrz_lines,
        }
    
    return result


def main() -> None:
    """Main entry point."""
    ap = argparse.ArgumentParser(
        description="INE obstruction detector v3 - robust detection using face/QR signatures"
    )
    ap.add_argument("input", help="Image file or folder")
    ap.add_argument("--debug-dir", default=None, help="Where to save debug overlays")
    ap.add_argument("--json-out", default=None, help="Write JSON report to file")
    ap.add_argument("--force-side", choices=["front", "back"], default=None,
                    help="Force side classification")
    ap.add_argument("--log-metrics", action="store_true",
                    help="Log detailed metrics for threshold tuning")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = ap.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    
    imgs = iter_images(input_path)
    if not imgs:
        raise SystemExit("No images found")
    
    results = []
    for p in imgs:
        r = process_one(
            p,
            debug_dir=debug_dir,
            force_side=args.force_side,
            log_metrics=args.log_metrics
        )
        results.append(r)
    
    # Summary
    ok = sum(1 for r in results if r.get("ok"))
    obs = sum(1 for r in results if r.get("ok") and r.get("obstructed"))
    print(f"Processed: {len(results)} | ok: {ok} | obstructed: {obs}")
    
    # List failures
    fails = [r for r in results if not r.get("ok")]
    if fails:
        print("Failures:")
        for r in fails[:10]:
            print(f"  - {r['path']}: {r.get('error')}")
    
    # Write JSON report
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
