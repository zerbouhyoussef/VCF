'''MCTF: Motion-Compensated Temporal Filtering with Hierarchical B-frames.

Implementation features:
- Bidirectional prediction for B-frames
- Hierarchical B-frame structure
- Integration with VCF spatial transforms (2D-DCT, 2D-DWT, etc.)
- Reuses VCF quantizers, color transforms, and entropy codecs
'''

import sys
import io
import os
import tempfile
import logging
import importlib
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2
import av
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# VCF Framework
with open(os.path.join(tempfile.gettempdir(), "description.txt"), 'w') as f:
    f.write(__doc__)

import main
import parser
import entropy_video_coding as EVC

# =============================================================================
# Constants and Configuration
# =============================================================================

class FrameType(Enum):
    I = "I"  # Intra frame
    P = "P"  # Predicted frame (forward only)
    B = "B"  # Bidirectional frame

@dataclass
class MotionVector:
    """Motion vector with cost information."""
    dx: int
    dy: int
    sad: float
    bits: float
    cost: float
    ref_idx: int  # Reference frame index

@dataclass
class EncodedFrame:
    """Frame encoding information."""
    frame_idx: int
    frame_type: FrameType
    display_order: int
    encoding_order: int
    references: List[int]  # Reference frame indices
    data: Optional[np.ndarray] = None
    mv_field: Optional[np.ndarray] = None
    residual: Optional[np.ndarray] = None

# =============================================================================
# Arguments
# =============================================================================

DEFAULT_ENCODE_OUTPUT_PREFIX = os.path.join(tempfile.gettempdir(), "encoded")
DEFAULT_DECODE_OUTPUT_PREFIX = os.path.join(tempfile.gettempdir(), "decoded")

# Encoder - MCTF-specific parameters only
# Note: -T (transform) is needed but not in VCF base parser, so we add it here
parser.parser_encode.add_argument("-T", "--transform", type=str,
    help=f"2D spatial transform for residuals (default: {EVC.DEFAULT_TRANSFORM})",
    default=EVC.DEFAULT_TRANSFORM)
parser.parser_encode.add_argument("-M", "--block_size_ME", type=parser.int_or_str,
    help="Block size for motion estimation (default: 16)",
    default=16)
parser.parser_encode.add_argument("-S", "--search_range", type=parser.int_or_str,
    help="Search range in pixels (default: 32)",
    default=32)
parser.parser_encode.add_argument("--fast", action="store_true",
    help="Use fast motion estimation (diamond search)")
parser.parser_encode.add_argument("--gop_size", type=int,
    help="GOP size (default: 16)",
    default=16)
parser.parser_encode.add_argument("--num_gops", type=int,
    help="Number of GOPs to encode (default: 1)",
    default=1)
parser.parser_encode.add_argument("--max_b_frames", type=int,
    help="Maximum consecutive B-frames (default: 10, minimum for prediction)",
    default=10)
parser.parser_encode.add_argument("--hierarchical", action="store_true",
    help="Use hierarchical B-frame structure")
parser.parser_encode.add_argument("--lambda_rd", type=float,
    help="Lagrange multiplier for RD optimization (default: 0.92*QSS). "
         "Higher = favor rate, lower = favor distortion. 0 = pure SAD.",
    default=None)

# Decoder - add transform parameter
parser.parser_decode.add_argument("-T", "--transform", type=str,
    help=f"2D spatial transform for residuals (default: {EVC.DEFAULT_TRANSFORM})",
    default=EVC.DEFAULT_TRANSFORM)

# Parse and import transform
args = parser.parser.parse_known_args()[0]

# Get transform name - use VCF's default if not specified
transform_name = getattr(args, 'transform', EVC.DEFAULT_TRANSFORM)

if __debug__:
    if args.debug:
        print(f"MCTF: Importing {transform_name}")

try:
    transform = importlib.import_module(transform_name)
except ImportError as e:
    print(f"Error: Could not find {transform_name} module ({e})")
    sys.exit(1)

# =============================================================================
# Motion Estimation Utilities
# =============================================================================

def compute_sad(block1: np.ndarray, block2: np.ndarray) -> float:
    """Compute Sum of Absolute Differences between two blocks."""
    return float(np.sum(np.abs(block1.astype(np.int16) - block2.astype(np.int16))))

def estimate_mv_bits(dx: int, dy: int) -> float:
    """Estimate bits to encode a motion vector using Exp-Golomb-like model.
    
    Each component costs  2·floor(log2(|v| + 1)) + 1  bits.
    Zero MV = 2 bits total (cheapest), large MV = expensive.
    """
    def _component_bits(v):
        return 2.0 * math.floor(math.log2(abs(v) + 1)) + 1.0
    return _component_bits(dx) + _component_bits(dy)

# Bits to signal prediction mode in a B-frame block:
#   fwd-only or bwd-only = 2 bits  (1 bit skip-bi + 1 bit direction)
#   bidirectional        = 3 bits  (1 bit skip-bi + 2 MVs flag)
MODE_BITS = {0: 2.0, 1: 2.0, 2: 3.0}

# =============================================================================
# Motion Estimation
# =============================================================================

def _diamond_search(ref_frame, curr_block, i, j, bs, sr=32, lmbda=0.0):
    """Diamond search pattern for motion estimation with RD cost.
    
    Minimises  J = SAD + λ · estimate_mv_bits(dx, dy).
    Returns (best_mv, best_sad)  — the raw SAD of the winner
    so callers can still use it for further mode decisions.
    """
    h, w = ref_frame.shape[:2]
    
    # Large diamond pattern
    ldp = [(0, -2), (-1, -1), (1, -1), (-2, 0), (2, 0), (-1, 1), (1, 1), (0, 2)]
    # Small diamond pattern
    sdp = [(0, -1), (-1, 0), (1, 0), (0, 1)]
    
    cx, cy = j, i
    best_mv = (0, 0)
    best_sad = float('inf')
    best_cost = float('inf')
    
    # Evaluate center position (MV = (0,0), cheapest rate)
    if 0 <= cy and cy + bs <= h and 0 <= cx and cx + bs <= w:
        best_sad = compute_sad(curr_block, ref_frame[cy:cy+bs, cx:cx+bs])
        best_cost = best_sad + lmbda * estimate_mv_bits(0, 0)
    
    # Large diamond search
    improved = True
    while improved:
        improved = False
        for ddx, ddy in ldp:
            ry, rx = cy + ddy, cx + ddx
            if (ry < 0 or ry + bs > h or rx < 0 or rx + bs > w or
                abs(rx - j) > sr or abs(ry - i) > sr):
                continue
            
            sad = compute_sad(curr_block, ref_frame[ry:ry+bs, rx:rx+bs])
            mv_dx, mv_dy = rx - j, ry - i
            cost = sad + lmbda * estimate_mv_bits(mv_dx, mv_dy)
            
            if cost < best_cost:
                best_sad = sad
                best_cost = cost
                best_mv = (mv_dx, mv_dy)
                cx, cy = rx, ry
                improved = True
    
    # Small diamond search refinement
    improved = True
    while improved:
        improved = False
        for ddx, ddy in sdp:
            ry, rx = cy + ddy, cx + ddx
            if (ry < 0 or ry + bs > h or rx < 0 or rx + bs > w or
                abs(rx - j) > sr or abs(ry - i) > sr):
                continue
            
            sad = compute_sad(curr_block, ref_frame[ry:ry+bs, rx:rx+bs])
            mv_dx, mv_dy = rx - j, ry - i
            cost = sad + lmbda * estimate_mv_bits(mv_dx, mv_dy)
            
            if cost < best_cost:
                best_sad = sad
                best_cost = cost
                best_mv = (mv_dx, mv_dy)
                cx, cy = rx, ry
                improved = True
    
    return best_mv, best_sad

def _exhaustive_search(ref_frame, curr_block, i, j, bs, sr=32, lmbda=0.0):
    """Exhaustive search for motion estimation with RD cost.
    
    Minimises  J = SAD + λ · estimate_mv_bits(dx, dy).
    """
    h, w = ref_frame.shape[:2]
    best_mv = (0, 0)
    best_sad = float('inf')
    best_cost = float('inf')
    
    for dy in range(-sr, sr + 1):
        for dx in range(-sr, sr + 1):
            ry, rx = i + dy, j + dx
            if 0 <= ry and ry + bs <= h and 0 <= rx and rx + bs <= w:
                sad = compute_sad(curr_block, ref_frame[ry:ry+bs, rx:rx+bs])
                cost = sad + lmbda * estimate_mv_bits(dx, dy)
                
                if cost < best_cost:
                    best_sad = sad
                    best_cost = cost
                    best_mv = (dx, dy)
    
    return best_mv, best_sad

def _process_block(args_tuple):
    """Process a single block for motion estimation (for parallel execution)."""
    ref, curr_block, i, j, bs, sr, fast, lmbda = args_tuple
    
    if fast:
        mv, sad = _diamond_search(ref, curr_block, i, j, bs, sr, lmbda)
    else:
        mv, sad = _exhaustive_search(ref, curr_block, i, j, bs, sr, lmbda)
    
    return mv, sad

def _process_row(args_tuple):
    """Process one row of blocks for motion estimation."""
    ref, curr, i, bs, sr, w, fast, lmbda = args_tuple
    mvs = []
    sads = []
    
    for j in range(0, w - bs + 1, bs):
        block = curr[i:i+bs, j:j+bs]
        
        if fast:
            mv, sad = _diamond_search(ref, block, i, j, bs, sr, lmbda)
        else:
            mv, sad = _exhaustive_search(ref, block, i, j, bs, sr, lmbda)
        
        mvs.append(mv)
        sads.append(sad)
    
    return mvs, sads

def block_matching(ref_frame, curr_frame, bs=16, sr=32, fast=True, lmbda=0.0):
    """Block-based motion estimation with RD-optimised MV selection.
    
    Each block minimises  J = SAD + λ · R_mv  instead of pure SAD.
    """
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    h, w = ref_gray.shape
    
    mv_h = (h - bs) // bs + 1
    mv_w = (w - bs) // bs + 1
    mv_field = np.zeros((mv_h, mv_w, 2), dtype=np.float32)
    
    # For exhaustive search, use multiprocessing; for fast search, use threading
    if not fast and sr > 16:
        # Exhaustive search with large search range - use multiprocessing
        from multiprocessing import Pool
        
        # Create block tasks
        block_args = []
        for i in range(0, h - bs + 1, bs):
            for j in range(0, w - bs + 1, bs):
                block = curr_gray[i:i+bs, j:j+bs]
                block_args.append((ref_gray, block, i, j, bs, sr, fast, lmbda))
        
        # Process blocks in parallel
        with Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(_process_block, block_args)
        
        # Reshape results into MV field
        idx = 0
        for ri in range(mv_h):
            for ci in range(mv_w):
                mv_field[ri, ci] = results[idx][0]
                idx += 1
    else:
        # Fast search or small search range - use threading (row-based)
        row_args = [(ref_gray, curr_gray, i, bs, sr, w, fast, lmbda) 
                    for i in range(0, h - bs + 1, bs)]
        
        with ThreadPoolExecutor(max_workers=max(1, mp.cpu_count() - 1)) as ex:
            results = list(ex.map(_process_row, row_args))
        
        for ri, (row_mvs, row_sads) in enumerate(results):
            for ci, mv in enumerate(row_mvs):
                mv_field[ri, ci] = mv
    
    return mv_field

# =============================================================================
# Bidirectional Motion Estimation
# =============================================================================

def _process_bidir_block(args_tuple):
    """Process bidirectional ME for a single block with RD-optimised mode decision.
    
    Mode decision minimises  J = SAD + λ · (mv_bits + mode_bits).
    """
    past_gray, future_gray, curr_block, i, j, bs, sr, fast, h, w, lmbda = args_tuple
    
    # Forward prediction (from past)
    if fast:
        mv_fwd, sad_fwd = _diamond_search(past_gray, curr_block, i, j, bs, sr, lmbda)
    else:
        mv_fwd, sad_fwd = _exhaustive_search(past_gray, curr_block, i, j, bs, sr, lmbda)
    
    # Backward prediction (from future)
    if fast:
        mv_bwd, sad_bwd = _diamond_search(future_gray, curr_block, i, j, bs, sr, lmbda)
    else:
        mv_bwd, sad_bwd = _exhaustive_search(future_gray, curr_block, i, j, bs, sr, lmbda)
    
    # Bidirectional (average of both predictions)
    ry_fwd = max(0, min(i + int(mv_fwd[1]), h - bs))
    rx_fwd = max(0, min(j + int(mv_fwd[0]), w - bs))
    ry_bwd = max(0, min(i + int(mv_bwd[1]), h - bs))
    rx_bwd = max(0, min(j + int(mv_bwd[0]), w - bs))
    
    pred_fwd = past_gray[ry_fwd:ry_fwd+bs, rx_fwd:rx_fwd+bs]
    pred_bwd = future_gray[ry_bwd:ry_bwd+bs, rx_bwd:rx_bwd+bs]
    pred_bi = ((pred_fwd.astype(np.int16) + pred_bwd.astype(np.int16)) // 2).astype(np.uint8)
    
    sad_bi = compute_sad(curr_block, pred_bi)
    
    # RD cost for each mode:  J = SAD + λ · (mv_bits + mode_bits)
    cost_fwd = sad_fwd + lmbda * (estimate_mv_bits(*mv_fwd) + MODE_BITS[0])
    cost_bwd = sad_bwd + lmbda * (estimate_mv_bits(*mv_bwd) + MODE_BITS[1])
    cost_bi  = sad_bi  + lmbda * (estimate_mv_bits(*mv_fwd) + estimate_mv_bits(*mv_bwd) + MODE_BITS[2])
    
    # Choose best mode based on RD cost
    if cost_fwd <= cost_bwd and cost_fwd <= cost_bi:
        mode = 0  # Forward
        mv_f = mv_fwd
        mv_b = (0, 0)
    elif cost_bwd <= cost_bi:
        mode = 1  # Backward
        mv_f = (0, 0)
        mv_b = mv_bwd
    else:
        mode = 2  # Bidirectional
        mv_f = mv_fwd
        mv_b = mv_bwd
    
    return mv_f, mv_b, mode

def bidirectional_me(ref_past, ref_future, curr_frame, bs=16, sr=32, fast=True, lmbda=0.0):
    """
    Bidirectional motion estimation for B-frames with RD-optimised mode decision.
    
    MV selection minimises  J = SAD + λ · R_mv.
    Mode decision minimises  J = SAD + λ · (R_mv + R_mode).
    Returns forward MV, backward MV, and best mode (forward/backward/bi).
    """
    h, w = curr_frame.shape[:2]
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    past_gray = cv2.cvtColor(ref_past, cv2.COLOR_RGB2GRAY)
    future_gray = cv2.cvtColor(ref_future, cv2.COLOR_RGB2GRAY)
    
    mv_h = (h - bs) // bs + 1
    mv_w = (w - bs) // bs + 1
    
    mv_forward = np.zeros((mv_h, mv_w, 2), dtype=np.float32)
    mv_backward = np.zeros((mv_h, mv_w, 2), dtype=np.float32)
    mode = np.zeros((mv_h, mv_w), dtype=np.uint8)  # 0=fwd, 1=bwd, 2=bi
    
    # For exhaustive search, use multiprocessing
    if not fast and sr > 16:
        from multiprocessing import Pool
        
        # Create block tasks
        block_args = []
        for i in range(0, h - bs + 1, bs):
            for j in range(0, w - bs + 1, bs):
                curr_block = curr_gray[i:i+bs, j:j+bs]
                block_args.append((past_gray, future_gray, curr_block, i, j, bs, sr, fast, h, w, lmbda))
        
        # Process blocks in parallel
        with Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(_process_bidir_block, block_args)
        
        # Reshape results
        idx = 0
        for ri in range(mv_h):
            for ci in range(mv_w):
                mv_forward[ri, ci] = results[idx][0]
                mv_backward[ri, ci] = results[idx][1]
                mode[ri, ci] = results[idx][2]
                idx += 1
    else:
        # Fast search - use sequential processing
        for i in range(0, h - bs + 1, bs):
            for j in range(0, w - bs + 1, bs):
                curr_block = curr_gray[i:i+bs, j:j+bs]
                
                # Forward prediction (from past)
                if fast:
                    mv_fwd, sad_fwd = _diamond_search(past_gray, curr_block, i, j, bs, sr, lmbda)
                else:
                    mv_fwd, sad_fwd = _exhaustive_search(past_gray, curr_block, i, j, bs, sr, lmbda)
                
                # Backward prediction (from future)
                if fast:
                    mv_bwd, sad_bwd = _diamond_search(future_gray, curr_block, i, j, bs, sr, lmbda)
                else:
                    mv_bwd, sad_bwd = _exhaustive_search(future_gray, curr_block, i, j, bs, sr, lmbda)
                
                # Bidirectional (average of both)
                ry_fwd = max(0, min(i + int(mv_fwd[1]), h - bs))
                rx_fwd = max(0, min(j + int(mv_fwd[0]), w - bs))
                ry_bwd = max(0, min(i + int(mv_bwd[1]), h - bs))
                rx_bwd = max(0, min(j + int(mv_bwd[0]), w - bs))
                
                pred_fwd = past_gray[ry_fwd:ry_fwd+bs, rx_fwd:rx_fwd+bs]
                pred_bwd = future_gray[ry_bwd:ry_bwd+bs, rx_bwd:rx_bwd+bs]
                pred_bi = ((pred_fwd.astype(np.int16) + pred_bwd.astype(np.int16)) // 2).astype(np.uint8)
                
                sad_bi = compute_sad(curr_block, pred_bi)
                
                # RD cost for each mode:  J = SAD + λ · (mv_bits + mode_bits)
                cost_fwd = sad_fwd + lmbda * (estimate_mv_bits(*mv_fwd) + MODE_BITS[0])
                cost_bwd = sad_bwd + lmbda * (estimate_mv_bits(*mv_bwd) + MODE_BITS[1])
                cost_bi  = sad_bi  + lmbda * (estimate_mv_bits(*mv_fwd) + estimate_mv_bits(*mv_bwd) + MODE_BITS[2])
                
                # Choose best mode based on RD cost
                ri, ci = i // bs, j // bs
                if cost_fwd <= cost_bwd and cost_fwd <= cost_bi:
                    mode[ri, ci] = 0  # Forward
                    mv_forward[ri, ci] = mv_fwd
                    mv_backward[ri, ci] = (0, 0)
                elif cost_bwd <= cost_bi:
                    mode[ri, ci] = 1  # Backward
                    mv_forward[ri, ci] = (0, 0)
                    mv_backward[ri, ci] = mv_bwd
                else:
                    mode[ri, ci] = 2  # Bidirectional
                    mv_forward[ri, ci] = mv_fwd
                    mv_backward[ri, ci] = mv_bwd
    
    return mv_forward, mv_backward, mode

# =============================================================================
# Motion Compensation
# =============================================================================

def motion_compensate(frame, mv_field, bs=16, direction=1):
    """Apply motion compensation (single reference)."""
    h, w = frame.shape[:2]
    comp = np.zeros_like(frame)
    
    for i in range(0, h - bs + 1, bs):
        for j in range(0, w - bs + 1, bs):
            mv = mv_field[i // bs, j // bs] * direction
            ry = int(np.clip(i + mv[1], 0, h - bs))
            rx = int(np.clip(j + mv[0], 0, w - bs))
            comp[i:i+bs, j:j+bs] = frame[ry:ry+bs, rx:rx+bs]
    
    # Handle boundaries
    if h % bs != 0:
        comp[-(h % bs):, :] = frame[-(h % bs):, :]
    if w % bs != 0:
        comp[:, -(w % bs):] = frame[:, -(w % bs):]
    
    return comp

def motion_compensate_bidirectional(ref_past, ref_future, mv_fwd, mv_bwd, 
                                   mode, bs=16):
    """Apply bidirectional motion compensation."""
    h, w = ref_past.shape[:2]
    comp = np.zeros_like(ref_past)
    
    for i in range(0, h - bs + 1, bs):
        for j in range(0, w - bs + 1, bs):
            ri, ci = i // bs, j // bs
            m = mode[ri, ci]
            
            if m == 0:  # Forward only
                mv = mv_fwd[ri, ci]
                ry = int(np.clip(i + mv[1], 0, h - bs))
                rx = int(np.clip(j + mv[0], 0, w - bs))
                comp[i:i+bs, j:j+bs] = ref_past[ry:ry+bs, rx:rx+bs]
            
            elif m == 1:  # Backward only
                mv = mv_bwd[ri, ci]
                ry = int(np.clip(i + mv[1], 0, h - bs))
                rx = int(np.clip(j + mv[0], 0, w - bs))
                comp[i:i+bs, j:j+bs] = ref_future[ry:ry+bs, rx:rx+bs]
            
            else:  # Bidirectional
                mv_f = mv_fwd[ri, ci]
                mv_b = mv_bwd[ri, ci]
                ry_f = int(np.clip(i + mv_f[1], 0, h - bs))
                rx_f = int(np.clip(j + mv_f[0], 0, w - bs))
                ry_b = int(np.clip(i + mv_b[1], 0, h - bs))
                rx_b = int(np.clip(j + mv_b[0], 0, w - bs))
                
                pred_f = ref_past[ry_f:ry_f+bs, rx_f:rx_f+bs]
                pred_b = ref_future[ry_b:ry_b+bs, rx_b:rx_b+bs]
                comp[i:i+bs, j:j+bs] = ((pred_f.astype(np.int16) + 
                                        pred_b.astype(np.int16)) // 2).astype(np.uint8)
    
    # Handle boundaries
    if h % bs != 0:
        comp[-(h % bs):, :] = ref_past[-(h % bs):, :]
    if w % bs != 0:
        comp[:, -(w % bs):] = ref_past[:, -(w % bs):]
    
    return comp

# =============================================================================
# GOP Structure Management
# =============================================================================

def create_hierarchical_gop(gop_size: int, max_b_frames: int = 3) -> List[EncodedFrame]:
    """
    Create hierarchical B-frame GOP structure.
    
    Example for GOP=8, max_b=3:
    Display:  0  1  2  3  4  5  6  7  8
    Type:     I  B  B  B  P  B  B  B  I
    Encoding: 0  4  2  1  3  8  6  5  7
    Level:    0  2  1  2  0  2  1  2  0
    """
    frames = []
    display_order = 0
    
    # I-frame at start
    frames.append(EncodedFrame(
        frame_idx=0,
        frame_type=FrameType.I,
        display_order=0,
        encoding_order=0,
        references=[]
    ))
    display_order += 1
    
    # P or I frame at end of GOP
    if gop_size > 1:
        frames.append(EncodedFrame(
            frame_idx=gop_size - 1,
            frame_type=FrameType.P,
            display_order=gop_size - 1,
            encoding_order=1,
            references=[0]
        ))
        
        # Hierarchical B-frames
        def add_hierarchical_b(start_ref, end_ref, level, encoding_order):
            if end_ref - start_ref <= 1:
                return encoding_order
            
            mid = (start_ref + end_ref) // 2
            frames.append(EncodedFrame(
                frame_idx=mid,
                frame_type=FrameType.B,
                display_order=mid,
                encoding_order=encoding_order,
                references=[start_ref, end_ref]
            ))
            encoding_order += 1
            
            if level < max_b_frames:
                encoding_order = add_hierarchical_b(start_ref, mid, level + 1, encoding_order)
                encoding_order = add_hierarchical_b(mid, end_ref, level + 1, encoding_order)
            
            return encoding_order
        
        add_hierarchical_b(0, gop_size - 1, 1, 2)
    
    # Sort by encoding order
    frames.sort(key=lambda x: x.encoding_order)
    
    return frames

def create_simple_gop(gop_size: int, max_b_frames: int = 3) -> List[EncodedFrame]:
    """
    Create simple IBBP GOP structure.
    
    Example for GOP=8, max_b=3:
    Display:  0  1  2  3  4  5  6  7  8
    Type:     I  B  B  B  P  B  B  B  I
    Encoding: 0  4  1  2  3  8  5  6  7
    """
    frames = []
    
    # I-frame at start
    frames.append(EncodedFrame(
        frame_idx=0,
        frame_type=FrameType.I,
        display_order=0,
        encoding_order=0,
        references=[]
    ))
    
    # P or I frame at end of GOP
    if gop_size > 1:
        anchor_pos = gop_size - 1
        frames.append(EncodedFrame(
            frame_idx=anchor_pos,
            frame_type=FrameType.P,
            display_order=anchor_pos,
            encoding_order=1,
            references=[0]  # Reference I-frame
        ))
        
        # All B-frames reference I and P
        encoding_order = 2
        for b_pos in range(1, anchor_pos):
            frames.append(EncodedFrame(
                frame_idx=b_pos,
                frame_type=FrameType.B,
                display_order=b_pos,
                encoding_order=encoding_order,
                references=[0, anchor_pos]  # All B-frames reference (I, P)
            ))
            encoding_order += 1
    
    # Sort by encoding order
    frames.sort(key=lambda x: x.encoding_order)
    
    return frames

# =============================================================================
# CoDec Class
# =============================================================================

class CoDec(EVC.CoDec):
    """MCTF Codec with hierarchical B-frames using VCF framework transforms."""

    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)
        self.transform_codec = transform.CoDec(args)
        logging.info(f"Using {args.transform} spatial transform for residuals")
        
        # Pass QSS to transform codec if available
        if hasattr(args, 'QSS'):
            self.transform_codec.args.QSS = args.QSS
            logging.info(f"Set transform codec QSS to {args.QSS}")
            # Recreate the quantizer with the correct QSS
            # (transform codec's __init__ already created it with possibly wrong QSS)
            try:
                from scalar_quantization.deadzone_quantization import Deadzone_Quantizer
                self.transform_codec.Q = Deadzone_Quantizer(Q_step=args.QSS, min_val=0, max_val=255)
                logging.info(f"Recreated quantizer with QSS={args.QSS}")
            except Exception as e:
                logging.warning(f"Could not recreate quantizer: {e}")
        
        # Monkey-patch transform codec methods to use self.args when called without arguments
        # This fixes VCF framework's encode()/decode() which call these without args
        # Also capture decoded output in case decode_write fails to persist to disk
        self._captured_decode_output = [None]
        original_encode_read = self.transform_codec.encode_read
        original_encode_write = self.transform_codec.encode_write
        original_decode_read = self.transform_codec.decode_read
        original_decode_write = self.transform_codec.decode_write
        
        def patched_encode_read(fn=None):
            if fn is None:
                fn = self.transform_codec.args.original
            return original_encode_read(fn)
        
        def patched_encode_write(codestream, fn=None):
            if fn is None:
                fn = self.transform_codec.args.encoded
            return original_encode_write(codestream, fn)
        
        def patched_decode_read(fn=None):
            if fn is None:
                fn = self.transform_codec.args.encoded
            return original_decode_read(fn)
        
        def patched_decode_write(img, fn=None):
            if fn is None:
                fn = self.transform_codec.args.decoded
            self._captured_decode_output[0] = np.array(img, copy=True)
            return original_decode_write(img, fn)
        
        self.transform_codec.encode_read = patched_encode_read
        self.transform_codec.encode_write = patched_encode_write
        self.transform_codec.decode_read = patched_decode_read
        self.transform_codec.decode_write = patched_decode_write
        
        self.block_size = int(getattr(args, 'block_size_ME', 16))
        self.search_range = int(getattr(args, 'search_range', 32))
        self.fast = getattr(args, 'fast', False)
        self.gop_size = int(getattr(args, 'gop_size', 16))
        self.num_gops = int(getattr(args, 'num_gops', 1))
        self.max_b_frames = int(getattr(args, 'max_b_frames', 10))
        self.hierarchical = getattr(args, 'hierarchical', False)
        
        # Ensure we have at least 10 frames for prediction
        if self.max_b_frames < 10:
            logging.warning(f"max_b_frames={self.max_b_frames} is too low, setting to 10")
            self.max_b_frames = 10
        
        # Where original frames are expected for metrics
        self.original_prefix = getattr(args, "original_prefix",
                                       os.path.join(tempfile.gettempdir(), "encoded_original"))
        
        # RD-optimization: λ controls the rate-distortion trade-off
        #   J = SAD + λ · R   (λ=0 ⟹ pure SAD, higher λ ⟹ favor cheaper MVs)
        # Default: √0.85 · QSS ≈ 0.92·QSS  (H.264 SAD-λ relationship)
        user_lambda = getattr(args, 'lambda_rd', None)
        if user_lambda is not None:
            self.lambda_rd = float(user_lambda)
        else:
            qss = float(getattr(args, 'QSS', 1))
            self.lambda_rd = math.sqrt(0.85) * qss
        
        logging.info(f"MCTF Config: GOP={self.gop_size}, NumGOPs={self.num_gops}, MaxB={self.max_b_frames}, "
                    f"BlockSize={self.block_size}, SearchRange={self.search_range}, "
                    f"Fast={self.fast}, Hierarchical={self.hierarchical}, "
                    f"λ_RD={self.lambda_rd:.4f}")

    def bye(self):
        """Override parent's bye() to prevent double video encoding."""
        logging.debug("trace")
        
        # Only calculate metrics without re-encoding
        if not self.encoding:
            logging.info("MCTF: Skipping VCF's automatic video re-encoding (already done)")

    def encode(self):
        """Encode video with MCTF."""
        logging.debug("trace")
        fn = self.args.original
        logging.info(f"Encoding {fn}")
        
        # Check if input is likely a video file
        if fn.endswith('.png') or fn.endswith('.jpg') or fn.endswith('.jpeg'):
            logging.error(f"MCTF requires a video file (e.g., .mp4, .avi) as input, not an image file.")
            logging.error(f"Received: {fn}")
            logging.error("Please use -o with a video file URL or path.")
            return 0
        
        try:
            container = av.open(fn)
        except Exception as e:
            logging.error(f"Cannot open video file {fn}: {e}")
            logging.error("MCTF requires a video file (e.g., .mp4, .avi) as input.")
            return 0
        
        # Calculate total frames from gop_size * num_gops
        total_frames_to_encode = self.gop_size * self.num_gops
        logging.info(f"Total frames to encode: {total_frames_to_encode} (GOP size={self.gop_size} × num GOPs={self.num_gops})")
        
        # Read frames
        frames = []
        for packet in container.demux():
            if __debug__:
                self.total_input_size += packet.size
            for frame in packet.decode():
                img = np.array(frame.to_image().convert("RGB"))
                frames.append(img)
                if len(frames) >= total_frames_to_encode:
                    break
            if len(frames) >= total_frames_to_encode:
                break
        container.close()
        
        if len(frames) < 2:
            logging.error("Need at least 2 frames")
            return 0
        
        self.N_frames = len(frames)
        self.height, self.width = frames[0].shape[:2]
        self.N_channels = 3
        # Force original frames to be saved in /tmp
        self.original_prefix = os.path.join(tempfile.gettempdir(), "encoded_original")
        
        logging.info(f"Video: {self.width}x{self.height}, {self.N_frames} frames, {self.N_channels} channels")
        
        # Save original frames
        for idx, img in enumerate(frames):
            img_fn = f"{self.original_prefix}_{idx:04d}.png"
            Image.fromarray(img).save(img_fn)
        
        # Process GOPs
        decoded_frames = {}  # Cache for reference frames
        
        for gop_start in range(0, self.N_frames, self.gop_size):
            gop_end = min(gop_start + self.gop_size, self.N_frames)
            actual_gop_size = gop_end - gop_start
            gop_start_size = self.total_output_size
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing GOP {gop_start}-{gop_end-1} (size={actual_gop_size})")
            logging.info(f"{'='*60}")
            
            # Create GOP structure
            if self.hierarchical:
                gop_structure = create_hierarchical_gop(actual_gop_size, self.max_b_frames)
            else:
                gop_structure = create_simple_gop(actual_gop_size, self.max_b_frames)
            
            # Encode frames in encoding order
            i_count = p_count = b_count = 0
            for frame_info in gop_structure:
                abs_idx = gop_start + frame_info.frame_idx
                if abs_idx >= self.N_frames:
                    continue
                
                frame = frames[abs_idx]
                
                if frame_info.frame_type == FrameType.I:
                    self._encode_i_frame(frame, abs_idx, decoded_frames)
                    i_count += 1
                
                elif frame_info.frame_type == FrameType.P:
                    ref_idx = gop_start + frame_info.references[0]
                    self._encode_p_frame(frame, abs_idx, ref_idx, decoded_frames)
                    p_count += 1
                
                elif frame_info.frame_type == FrameType.B:
                    ref_past_idx = gop_start + frame_info.references[0]
                    ref_future_idx = gop_start + frame_info.references[1]
                    self._encode_b_frame(frame, abs_idx, ref_past_idx, 
                                       ref_future_idx, decoded_frames)
                    b_count += 1
            
            # GOP statistics
            gop_bytes = self.total_output_size - gop_start_size
            gop_pixels = actual_gop_size * self.width * self.height
            gop_bpp = (gop_bytes * 8) / gop_pixels if gop_pixels > 0 else 0
            
            logging.info(f"\nGOP {gop_start}-{gop_end-1} Summary:")
            logging.info(f"  Frame types: I={i_count}, P={p_count}, B={b_count}")
            logging.info(f"  GOP size: {gop_bytes} bytes ({gop_bpp:.4f} bpp)")
            logging.info(f"  Average: {gop_bytes/actual_gop_size:.2f} bytes/frame")
        
        logging.info(f"\n{'='*60}")
        logging.info(f"ENCODING COMPLETE")
        logging.info(f"{'='*60}")
        
        # Calculate and log metrics
        total_pixels = self.N_frames * self.width * self.height
        BPP = (self.total_output_size * 8) / total_pixels
        
        logging.info(f"Total frames encoded: {self.N_frames}")
        logging.info(f"Video dimensions: {self.width}x{self.height}")
        logging.info(f"Total output size: {self.total_output_size} bytes ({self.total_output_size/1024/1024:.2f} MB)")
        logging.info(f"Total input size: {self.total_input_size} bytes ({self.total_input_size/1024/1024:.2f} MB)")
        logging.info(f"Compression ratio: {self.total_input_size/self.total_output_size:.2f}:1")
        logging.info(f"Total pixels: {total_pixels}")
        logging.info(f"Bits Per Pixel (BPP): {BPP:.6f}")
        logging.info(f"Compression rate: {BPP:.4f} bits/pixel")
        logging.info(f"\nNOTE: The encoded size ({self.total_output_size/1024/1024:.2f} MB) is the actual compressed data.")
        logging.info(f"The decoded MP4 is for playback only and uses H.264 re-encoding.")
        logging.info(f"{'='*60}\n")
        
        # Persist metadata so decode() is self-contained
        with open(f"{self.args.encoded}_meta.txt", "w") as f:
            f.write(f"{self.original_prefix}\n")
            f.write(f"{self.N_frames}\n")
            f.write(f"{self.height}\n")
            f.write(f"{self.width}\n")
            f.write(f"{BPP}\n")
            f.write(f"{self.total_output_size}\n")
        
        return self.total_output_size

    def _encode_i_frame(self, frame, idx, decoded_frames):
        """Encode I-frame."""
        logging.info(f"Encoding I-frame {idx}")
        
        i_orig_fn = f"{self.args.encoded}_I_{idx:04d}.png"
        i_enc_fn = f"{self.args.encoded}_{idx:04d}"
        Image.fromarray(frame).save(i_orig_fn)
        
        # Temporarily set args so transform codec's encode() reads/writes correct files
        saved_original = getattr(self.transform_codec.args, 'original', None)
        saved_encoded = getattr(self.transform_codec.args, 'encoded', None)
        
        self.transform_codec.args.original = i_orig_fn
        self.transform_codec.args.encoded = i_enc_fn
        
        # Recreate quantizer with correct QSS before encoding
        if hasattr(self.args, 'QSS'):
            self.transform_codec.args.QSS = self.args.QSS
            from scalar_quantization.deadzone_quantization import Deadzone_Quantizer
            self.transform_codec.Q = Deadzone_Quantizer(Q_step=self.args.QSS, min_val=0, max_val=255)
        
        # Use full encode() to ensure DCT transform and quantization are applied
        O_bytes = self.transform_codec.encode()
        
        # Restore args
        if saved_original is not None:
            self.transform_codec.args.original = saved_original
        if saved_encoded is not None:
            self.transform_codec.args.encoded = saved_encoded
        
        self.total_output_size += O_bytes
        
        # Save metadata
        with open(f"{i_enc_fn}_type.txt", 'w') as f:
            f.write("I")
        
        # Decode and cache for references
        dec_fn = f"{self.args.encoded}_dec_{idx:04d}.png"
        saved_encoded = getattr(self.transform_codec.args, 'encoded', None)
        saved_decoded = getattr(self.transform_codec.args, 'decoded', None)
        
        self.transform_codec.args.encoded = i_enc_fn
        self.transform_codec.args.decoded = dec_fn
        self._captured_decode_output[0] = None
        self.transform_codec.decode()
        
        if saved_encoded is not None:
            self.transform_codec.args.encoded = saved_encoded
        if saved_decoded is not None:
            self.transform_codec.args.decoded = saved_decoded
        
        # Use captured output if available (avoids FileNotFoundError when decode_write path differs)
        if self._captured_decode_output[0] is not None:
            decoded_frames[idx] = self._captured_decode_output[0]
        elif os.path.exists(dec_fn):
            decoded_frames[idx] = np.array(Image.open(dec_fn).convert("RGB"))
        else:
            raise FileNotFoundError(
                f"Decoded frame not found at {dec_fn}. "
                "The transform codec's decode() did not write the expected output."
            )
        
        logging.info(f"  I-frame {idx}: {O_bytes} bytes")

    def _encode_p_frame(self, frame, idx, ref_idx, decoded_frames):
        """Encode P-frame (forward prediction only) using VCF transform."""
        logging.info(f"Encoding P-frame {idx} (ref={ref_idx})")
        
        ref_frame = decoded_frames[ref_idx]
        
        # Motion estimation (RD-optimised: J = SAD + λ·R_mv)
        mv_field = block_matching(
            ref_frame, frame, 
            self.block_size, self.search_range, 
            self.fast, self.lambda_rd
        )
        
        # Motion compensation
        pred = motion_compensate(ref_frame, mv_field, self.block_size)
        
        # Compute residual and map to [0, 255] without hard-clipping:
        # residual ∈ [-255, 255] → /2 + 128 → [0.5, 255.5] → round → [0, 255]
        # Max rounding error from mapping: ±1 per sample (vs up to ±128 with old +128 clip)
        residual = frame.astype(np.int16) - pred.astype(np.int16)
        residual_img = np.clip(np.round(residual.astype(np.float32) / 2 + 128), 0, 255).astype(np.uint8)
        
        # Save residual as PNG and encode using transform codec (DCT + quantize + entropy)
        residual_fn = f"{self.args.encoded}_res_{idx:04d}.png"
        Image.fromarray(residual_img).save(residual_fn)
        
        enc_fn = f"{self.args.encoded}_{idx:04d}"
        
        # Temporarily set args
        saved_original = getattr(self.transform_codec.args, 'original', None)
        saved_encoded = getattr(self.transform_codec.args, 'encoded', None)
        
        self.transform_codec.args.original = residual_fn
        self.transform_codec.args.encoded = enc_fn
        
        # Recreate quantizer before encoding
        if hasattr(self.args, 'QSS'):
            self.transform_codec.args.QSS = self.args.QSS
            from scalar_quantization.deadzone_quantization import Deadzone_Quantizer
            self.transform_codec.Q = Deadzone_Quantizer(Q_step=self.args.QSS, min_val=0, max_val=255)
        
        O_bytes = self.transform_codec.encode()
        
        # Restore args
        if saved_original is not None:
            self.transform_codec.args.original = saved_original
        if saved_encoded is not None:
            self.transform_codec.args.encoded = saved_encoded
        
        self.total_output_size += O_bytes
        
        # Save motion vectors
        np.savez_compressed(
            f"{enc_fn}_mv.npz",
            mv=mv_field,
            ref_idx=ref_idx
        )
        mv_size = os.path.getsize(f"{enc_fn}_mv.npz")
        self.total_output_size += mv_size
        
        # Save frame type
        with open(f"{enc_fn}_type.txt", 'w') as f:
            f.write(f"P:{ref_idx}")
        
        # Decode residual to get reconstruction (encoder-side decode for reference)
        dec_fn = f"{self.args.encoded}_dec_{idx:04d}.png"
        saved_encoded = getattr(self.transform_codec.args, 'encoded', None)
        saved_decoded = getattr(self.transform_codec.args, 'decoded', None)
        
        self.transform_codec.args.encoded = enc_fn
        self.transform_codec.args.decoded = dec_fn
        self._captured_decode_output[0] = None
        self.transform_codec.decode()
        
        self.transform_codec.args.encoded = saved_encoded
        self.transform_codec.args.decoded = saved_decoded
        
        # Undo mapping: (pixel - 128) * 2; use captured output if available
        if self._captured_decode_output[0] is not None:
            residual_img_dec = self._captured_decode_output[0]
        elif os.path.exists(dec_fn):
            residual_img_dec = np.array(Image.open(dec_fn).convert("RGB"))
        else:
            raise FileNotFoundError(
                f"Decoded residual not found at {dec_fn}. "
                "The transform codec's decode() did not write the expected output."
            )
        residual_rec = (residual_img_dec.astype(np.int16) - 128) * 2
        
        # Reconstruct and cache
        recon = np.clip(pred.astype(np.int16) + residual_rec, 0, 255).astype(np.uint8)
        decoded_frames[idx] = recon
        
        logging.info(f"  P-frame {idx}: residual={O_bytes} bytes, mv={mv_size} bytes")

    def _encode_b_frame(self, frame, idx, ref_past_idx, ref_future_idx, decoded_frames):
        """Encode B-frame (bidirectional prediction) using VCF transform."""
        logging.info(f"Encoding B-frame {idx} (refs={ref_past_idx},{ref_future_idx})")
        
        ref_past = decoded_frames[ref_past_idx]
        ref_future = decoded_frames[ref_future_idx]
        
        # Bidirectional motion estimation (RD-optimised: J = SAD + λ·(R_mv + R_mode))
        mv_fwd, mv_bwd, mode = bidirectional_me(
            ref_past, ref_future, frame,
            self.block_size, self.search_range,
            self.fast, self.lambda_rd
        )
        
        # Motion compensation
        pred = motion_compensate_bidirectional(
            ref_past, ref_future, mv_fwd, mv_bwd, mode, self.block_size
        )
        
        # Compute residual and map to [0, 255] without hard-clipping:
        # residual ∈ [-255, 255] → /2 + 128 → [0.5, 255.5] → round → [0, 255]
        residual = frame.astype(np.int16) - pred.astype(np.int16)
        residual_img = np.clip(np.round(residual.astype(np.float32) / 2 + 128), 0, 255).astype(np.uint8)
        
        # Save residual as PNG and encode using transform codec (DCT + quantize + entropy)
        residual_fn = f"{self.args.encoded}_res_{idx:04d}.png"
        Image.fromarray(residual_img).save(residual_fn)
        
        enc_fn = f"{self.args.encoded}_{idx:04d}"
        
        # Temporarily set args
        saved_original = getattr(self.transform_codec.args, 'original', None)
        saved_encoded = getattr(self.transform_codec.args, 'encoded', None)
        
        self.transform_codec.args.original = residual_fn
        self.transform_codec.args.encoded = enc_fn
        
        # Recreate quantizer before encoding
        if hasattr(self.args, 'QSS'):
            self.transform_codec.args.QSS = self.args.QSS
            from scalar_quantization.deadzone_quantization import Deadzone_Quantizer
            self.transform_codec.Q = Deadzone_Quantizer(Q_step=self.args.QSS, min_val=0, max_val=255)
        
        O_bytes = self.transform_codec.encode()
        
        # Restore args
        if saved_original is not None:
            self.transform_codec.args.original = saved_original
        if saved_encoded is not None:
            self.transform_codec.args.encoded = saved_encoded
        
        self.total_output_size += O_bytes
        
        # Save motion vectors and mode
        np.savez_compressed(
            f"{enc_fn}_mv.npz",
            mv_fwd=mv_fwd,
            mv_bwd=mv_bwd,
            mode=mode,
            ref_past=ref_past_idx,
            ref_future=ref_future_idx
        )
        mv_size = os.path.getsize(f"{enc_fn}_mv.npz")
        self.total_output_size += mv_size
        
        # Save frame type
        with open(f"{enc_fn}_type.txt", 'w') as f:
            f.write(f"B:{ref_past_idx},{ref_future_idx}")
        
        # Decode residual to get reconstruction (encoder-side decode for reference)
        dec_fn = f"{self.args.encoded}_dec_{idx:04d}.png"
        saved_encoded = getattr(self.transform_codec.args, 'encoded', None)
        saved_decoded = getattr(self.transform_codec.args, 'decoded', None)
        
        self.transform_codec.args.encoded = enc_fn
        self.transform_codec.args.decoded = dec_fn
        self._captured_decode_output[0] = None
        self.transform_codec.decode()
        
        self.transform_codec.args.encoded = saved_encoded
        self.transform_codec.args.decoded = saved_decoded
        
        # Undo mapping: (pixel - 128) * 2; use captured output if available
        if self._captured_decode_output[0] is not None:
            residual_img_dec = self._captured_decode_output[0]
        elif os.path.exists(dec_fn):
            residual_img_dec = np.array(Image.open(dec_fn).convert("RGB"))
        else:
            raise FileNotFoundError(
                f"Decoded residual not found at {dec_fn}. "
                "The transform codec's decode() did not write the expected output."
            )
        residual_rec = (residual_img_dec.astype(np.int16) - 128) * 2
        
        # Reconstruct and cache
        recon = np.clip(pred.astype(np.int16) + residual_rec, 0, 255).astype(np.uint8)
        decoded_frames[idx] = recon
        
        mode_stats = [np.sum(mode == i) for i in range(3)]
        logging.info(f"  B-frame {idx}: residual={O_bytes} bytes, mv={mv_size} bytes, "
                    f"modes(fwd/bwd/bi)={mode_stats}")

    def decode(self):
        """Decode MCTF encoded video."""
        logging.debug("trace")
        
        # Read encoding metadata if available (makes decode self-contained)
        meta = f"{self.args.encoded}_meta.txt"
        if os.path.exists(meta):
            with open(meta, "r") as f:
                self.original_prefix = f.readline().strip()
                self.N_frames = int(f.readline().strip())
                self.height = int(f.readline().strip())
                self.width = int(f.readline().strip())
                self._meta_BPP = float(f.readline().strip())
                line = f.readline().strip()
                if line:
                    self.total_output_size = int(line)
        
        # First, scan all frames to determine decoding order
        frame_info = {}
        idx = 0
        while True:
            type_fn = f"{self.args.encoded}_{idx:04d}_type.txt"
            
            if not os.path.exists(type_fn):
                if idx == 0:
                    logging.error("No encoded frames found")
                    return 0
                break
            
            with open(type_fn, 'r') as f:
                frame_type = f.read().strip()
            
            frame_info[idx] = {
                'type': frame_type,
                'display_order': idx
            }
            idx += 1
        
        total_frames = len(frame_info)
        logging.info(f"Found {total_frames} encoded frames")
        
        # Decode frames in dependency order (handles hierarchical B-frames)
        # Keep decoding frames whose references are ready
        decoded_frames = {}
        remaining_frames = set(range(total_frames))
        
        while remaining_frames:
            progress = False
            
            for idx in list(remaining_frames):
                frame_type = frame_info[idx]['type']
                can_decode = False
                
                if frame_type == "I":
                    # I-frames have no dependencies
                    can_decode = True
                
                elif frame_type.startswith("P:"):
                    # P-frames depend on one reference
                    ref_idx = int(frame_type.split(":")[1])
                    can_decode = ref_idx in decoded_frames
                
                elif frame_type.startswith("B:"):
                    # B-frames depend on two references
                    refs = frame_type.split(":")[1].split(",")
                    ref_past = int(refs[0])
                    ref_future = int(refs[1])
                    can_decode = (ref_past in decoded_frames and ref_future in decoded_frames)
                
                # Decode if ready
                if can_decode:
                    if frame_type == "I":
                        self._decode_i_frame(idx, decoded_frames)
                    elif frame_type.startswith("P:"):
                        ref_idx = int(frame_type.split(":")[1])
                        self._decode_p_frame(idx, ref_idx, decoded_frames)
                    elif frame_type.startswith("B:"):
                        refs = frame_type.split(":")[1].split(",")
                        ref_past = int(refs[0])
                        ref_future = int(refs[1])
                        self._decode_b_frame(idx, ref_past, ref_future, decoded_frames)
                    
                    remaining_frames.remove(idx)
                    progress = True
            
            # Check for deadlock
            if not progress:
                raise RuntimeError(f"Cannot decode remaining frames {remaining_frames} - "
                                   f"circular dependency or missing references")
        
        logging.info(f"\n{'='*60}")
        logging.info(f"DECODING COMPLETE")
        logging.info(f"{'='*60}")
        logging.info(f"Total frames decoded: {len(decoded_frames)}")
        
        # Write all decoded frames to disk with consistent naming
        # (guarantees files exist at the expected paths for RMSE and ffmpeg)
        decoded_prefix = getattr(self.args, 'decoded', '/tmp/decoded')
        if decoded_prefix.endswith('.png'):
            decoded_prefix = decoded_prefix[:-4]
        for idx in range(len(decoded_frames)):
            out_fn = f"{decoded_prefix}_{idx:04d}.png"
            Image.fromarray(decoded_frames[idx]).save(out_fn)
        logging.info(f"Wrote {len(decoded_frames)} frames to {decoded_prefix}_XXXX.png")
        
        # Calculate quality metrics (RMSE) if original frames are available
        try:
            from information_theory import distortion
            
            total_RMSE = 0
            frames_compared = 0
            
            for idx in range(len(decoded_frames)):
                original_fn = f"{self.original_prefix}_{idx:04d}.png"
                decoded_fn = f"{decoded_prefix}_{idx:04d}.png"
                
                if os.path.exists(original_fn) and os.path.exists(decoded_fn):
                    original_img = np.array(Image.open(original_fn).convert("RGB"))
                    decoded_img = np.array(Image.open(decoded_fn).convert("RGB"))
                    
                    frame_RMSE = distortion.RMSE(original_img, decoded_img)
                    total_RMSE += frame_RMSE
                    frames_compared += 1
                    
                    if idx < 3 or idx == len(decoded_frames) - 1:  # Log first 3 and last frame
                        logging.info(f"  Frame {idx} RMSE: {frame_RMSE:.4f}")
            
            if frames_compared > 0:
                avg_RMSE = total_RMSE / frames_compared
                
                # Calculate BPP from encoding metadata
                BPP = 0
                if hasattr(self, 'total_output_size') and self.total_output_size > 0 and hasattr(self, 'width') and hasattr(self, 'height'):
                    total_pixels = frames_compared * self.width * self.height
                    BPP = (self.total_output_size * 8) / total_pixels
                elif hasattr(self, '_meta_BPP') and self._meta_BPP > 0:
                    BPP = self._meta_BPP
                else:
                    logging.warning("Could not determine BPP from encoding metadata")
                
                lrd = getattr(self, 'lambda_rd', 1.0)
                J = avg_RMSE + lrd * BPP
                
                logging.info(f"\n{'='*60}")
                logging.info(f"QUALITY METRICS")
                logging.info(f"{'='*60}")
                logging.info(f"Frames compared: {frames_compared}")
                logging.info(f"Average RMSE (D): {avg_RMSE:.6f}")
                logging.info(f"Bits Per Pixel (R): {BPP:.6f}")
                logging.info(f"λ (lambda_rd): {lrd:.4f}")
                logging.info(f"Rate-Distortion Cost (J = D + λ·R): {J:.6f}")
                logging.info(f"{'='*60}\n")
        
        except ImportError:
            logging.warning("information_theory module not available, skipping RMSE calculation")
        except Exception as e:
            logging.warning(f"Error calculating metrics: {e}")
        
        # Create output video
        logging.info("Creating output video...")
        
        # Use ffmpeg to combine frames with optimized settings
        import subprocess
        try:
            # Verify decoded frames actually exist before calling ffmpeg
            first_frame = f"{decoded_prefix}_0000.png"
            logging.info(f"Decoded prefix: {decoded_prefix}")
            logging.info(f"First frame exists? {os.path.exists(first_frame)}")
            if not os.path.exists(first_frame):
                logging.error(f"Cannot create video: first decoded frame not found at {first_frame}")
                return 0
            
            output_mp4 = f'{decoded_prefix}.mp4'
            cmd = [
                'ffmpeg', '-y',
                '-framerate', '30',
                '-i', f'{decoded_prefix}_%04d.png',
                '-c:v', 'libx264',
                '-crf', '18',  # Near-lossless quality (0=lossless, 51=worst, 18=visually lossless)
                '-preset', 'medium',  # Encoding speed (slower = better compression)
                '-pix_fmt', 'yuv420p',
                output_mp4
            ]
            
            # Debug: show command
            logging.info(f"Running ffmpeg command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Get output video size
            if os.path.exists(output_mp4):
                mp4_size = os.path.getsize(output_mp4)
                logging.info(f"Video saved to {output_mp4}")
                logging.info(f"Output MP4 size: {mp4_size} bytes ({mp4_size/1024/1024:.2f} MB)")
                
                # Compare with encoded size (if available from encoding metadata)
                if hasattr(self, 'total_output_size') and self.total_output_size > 0:
                    ratio = mp4_size / self.total_output_size
                    logging.info(f"MP4 vs Encoded ratio: {ratio:.2f}x")
                    if ratio > 2:
                        logging.warning(f"MP4 is {ratio:.2f}x larger than encoded data!")
                else:
                    # Try to use metadata loaded at start of decode()
                    if hasattr(self, '_meta_BPP') and self._meta_BPP > 0 and hasattr(self, 'width') and hasattr(self, 'height') and hasattr(self, 'N_frames'):
                        total_pixels = self.N_frames * self.height * self.width
                        encoded_size = int((self._meta_BPP * total_pixels) / 8)
                        if encoded_size > 0:
                            ratio = mp4_size / encoded_size
                            logging.info(f"Encoded data size: {encoded_size} bytes ({encoded_size/1024/1024:.2f} MB)")
                            logging.info(f"MP4 vs Encoded ratio: {ratio:.2f}x")
                            if ratio > 2:
                                logging.info(f"NOTE: MP4 is {ratio:.2f}x larger - this is normal for H.264 re-encoding")
                    else:
                        logging.debug("Could not calculate encoded size for comparison")
            
            # Show ffmpeg output if debugging
            if result.stderr:
                logging.debug(f"FFmpeg output: {result.stderr}")
                
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to create video: {e}")
            logging.error(f"FFmpeg stderr: {e.stderr}")
            logging.error(f"FFmpeg stdout: {e.stdout}")
        except FileNotFoundError:
            logging.warning("ffmpeg not found, skipping video creation")
        
        return 0

    def _decode_i_frame(self, idx, decoded_frames):
        """Decode I-frame."""
        enc_fn = f"{self.args.encoded}_{idx:04d}"
        dec_fn = f"{getattr(self.args, 'decoded', '/tmp/decoded')}_{idx:04d}.png"
        
        logging.info(f"Decoding I-frame {idx}")
        saved_encoded = self.transform_codec.args.encoded
        saved_decoded = self.transform_codec.args.decoded
        
        self.transform_codec.args.encoded = enc_fn
        self.transform_codec.args.decoded = dec_fn
        self.transform_codec.decode()
        
        self.transform_codec.args.encoded = saved_encoded
        self.transform_codec.args.decoded = saved_decoded
        
        img = np.array(Image.open(dec_fn).convert("RGB"))
        decoded_frames[idx] = img

    def _decode_p_frame(self, idx, ref_idx, decoded_frames):
        """Decode P-frame using VCF transform."""
        logging.info(f"Decoding P-frame {idx} (ref={ref_idx})")
        
        enc_fn = f"{self.args.encoded}_{idx:04d}"
        
        # Load motion vectors
        mv_data = np.load(f"{enc_fn}_mv.npz")
        mv_field = mv_data['mv']
        
        # Decode residual using transform codec
        residual_fn = f"{self.args.encoded}_dec_res_{idx:04d}.png"
        saved_encoded = self.transform_codec.args.encoded
        saved_decoded = self.transform_codec.args.decoded
        
        self.transform_codec.args.encoded = enc_fn
        self.transform_codec.args.decoded = residual_fn
        self.transform_codec.decode()
        
        self.transform_codec.args.encoded = saved_encoded
        self.transform_codec.args.decoded = saved_decoded
        
        # Undo /2+128 mapping: (pixel - 128) * 2
        residual = (np.array(Image.open(residual_fn).convert("RGB")).astype(np.int16) - 128) * 2
        
        # Motion compensation
        ref_frame = decoded_frames[ref_idx]
        pred = motion_compensate(ref_frame, mv_field, self.block_size)
        
        # Reconstruct
        recon = np.clip(pred.astype(np.int16) + residual, 0, 255).astype(np.uint8)
        decoded_frames[idx] = recon

    def _decode_b_frame(self, idx, ref_past_idx, ref_future_idx, decoded_frames):
        """Decode B-frame using VCF transform."""
        logging.info(f"Decoding B-frame {idx} (refs={ref_past_idx},{ref_future_idx})")
        
        enc_fn = f"{self.args.encoded}_{idx:04d}"
        
        # Load motion vectors and mode
        mv_data = np.load(f"{enc_fn}_mv.npz")
        mv_fwd = mv_data['mv_fwd']
        mv_bwd = mv_data['mv_bwd']
        mode = mv_data['mode']
        
        # Decode residual using transform codec
        residual_fn = f"{self.args.encoded}_dec_res_{idx:04d}.png"
        saved_encoded = self.transform_codec.args.encoded
        saved_decoded = self.transform_codec.args.decoded
        
        self.transform_codec.args.encoded = enc_fn
        self.transform_codec.args.decoded = residual_fn
        self.transform_codec.decode()
        
        self.transform_codec.args.encoded = saved_encoded
        self.transform_codec.args.decoded = saved_decoded
        
        # Undo /2+128 mapping: (pixel - 128) * 2
        residual = (np.array(Image.open(residual_fn).convert("RGB")).astype(np.int16) - 128) * 2
        
        # Motion compensation
        ref_past = decoded_frames[ref_past_idx]
        ref_future = decoded_frames[ref_future_idx]
        pred = motion_compensate_bidirectional(
            ref_past, ref_future, mv_fwd, mv_bwd, mode, self.block_size
        )
        
        # Reconstruct
        recon = np.clip(pred.astype(np.int16) + residual, 0, 255).astype(np.uint8)
        decoded_frames[idx] = recon

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)