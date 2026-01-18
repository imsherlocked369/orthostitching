#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV fast-stitching with OpenCV ORB/AKAZE only (no neural backends).

Additions in this version:
- Pixel-wise overlap governor with coverage cap (default 3)
- Optional sharper-override to replace capped pixels only if new view is sharper
- Keeps: grid-uniform features, similarity model (nadir-ish), adaptive FROI,
         Laplacian/feather blending, RAM guard, optional auto-rotate
"""

from __future__ import annotations
import os, sys, math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ----------------- small utils -----------------
@dataclass
class Frame:
    path: Path
    bgr: np.ndarray
    s_match: float  # scale used for matching (full-res kept for warping)

def load_frames(img_dir: Path, match_max_side=1600) -> List[Frame]:
    exts = {".jpg",".jpeg",".png",".tif",".tiff",".bmp",
            ".JPG",".JPEG",".PNG",".TIF",".TIFF",".BMP"}
    files = sorted([p for p in img_dir.iterdir() if p.suffix in exts])
    frames: List[Frame] = []
    for p in files:
        arr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            print(f"[warn] failed to read {p.name}, skipping")
            continue
        H,W = arr.shape[:2]
        s = 1.0
        if max(H,W) > match_max_side:
            s = match_max_side/float(max(H,W))
        frames.append(Frame(p, arr, s))
    print(f"Found {len(frames)} images.")
    return frames


# ----------------- keypoint helpers -----------------
def grid_uniform_kp(kps, desc, size, grid=(8,6), max_per_cell=500):
    if not kps:
        return np.empty((0,2), np.float32), None

    H, W = size
    gx, gy = grid
    cell_w = max(1, W // gx)
    cell_h = max(1, H // gy)

    buckets = [[[] for _ in range(gx)] for __ in range(gy)]
    for i, kp in enumerate(kps):
        x, y = kp.pt
        cx = min(gx-1, max(0, int(x // cell_w)))
        cy = min(gy-1, max(0, int(y // cell_h)))
        buckets[cy][cx].append(i)

    keep_idx = []
    for cy in range(gy):
        for cx in range(gx):
            idxs = buckets[cy][cx]
            if not idxs: 
                continue
            idxs.sort(key=lambda i: kps[i].response if hasattr(kps[i], "response") else 0.0,
                      reverse=True)
            keep_idx.extend(idxs[:max_per_cell])

    keep_idx = np.array(keep_idx, dtype=np.int32)
    pts = np.array([kps[i].pt for i in keep_idx], dtype=np.float32)
    dsc = desc[keep_idx] if desc is not None and len(desc)>0 else None
    return pts, dsc


def knn_ratio_and_mutual(d0, d1, norm, ratio=0.72) -> List[cv2.DMatch]:
    bf = cv2.BFMatcher(norm, crossCheck=False)
    m01 = bf.knnMatch(d0, d1, k=2)
    m10 = bf.knnMatch(d1, d0, k=2)

    def _best_map(matches):
        best = {}
        for pair in matches:
            if len(pair) < 1:
                continue
            m = pair[0]
            if len(pair) == 2 and pair[0].distance >= ratio * pair[1].distance:
                continue
            if m.queryIdx not in best or m.distance < best[m.queryIdx].distance:
                best[m.queryIdx] = m
        return best

    best01 = _best_map(m01)
    best10 = _best_map(m10)

    good = []
    for q0, m in best01.items():
        q1 = m.trainIdx
        m_back = best10.get(q1, None)
        if m_back is not None and m_back.trainIdx == q0:
            good.append(m)
    return good


# ----------------- OpenCV Matcher Backend -----------------
class MatcherBackend:
    def __init__(self, matcher_type="orb", max_features=5000, grid=(10,8), per_cell=300):
        self.mode = matcher_type
        self.grid = grid
        self.per_cell = per_cell
        self.norm = cv2.NORM_HAMMING

        if self.mode in ("orb", "hybrid"):
            self.orb = cv2.ORB_create(nfeatures=max_features)
        else:
            self.orb = None

        if self.mode in ("akaze", "hybrid"):
            self.akaze = cv2.AKAZE_create()
        else:
            self.akaze = None

    def _extract(self, im, which="orb"):
        if which == "orb":
            k, d = self.orb.detectAndCompute(im, None)
        else:
            k, d = self.akaze.detectAndCompute(im, None)
        if k is None or d is None or len(k) == 0 or len(d) == 0:
            return np.empty((0,2), np.float32), None
        pts, dsc = grid_uniform_kp(k, d, im.shape[:2], grid=self.grid, max_per_cell=self.per_cell)
        return pts, dsc

    def match(self, im0: np.ndarray, im1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.mode == "orb":
            p0, d0 = self._extract(im0, "orb")
            p1, d1 = self._extract(im1, "orb")
        elif self.mode == "akaze":
            p0, d0 = self._extract(im0, "akaze")
            p1, d1 = self._extract(im1, "akaze")
        else:
            p0o, d0o = self._extract(im0, "orb")
            p1o, d1o = self._extract(im1, "orb")
            p0a, d0a = self._extract(im0, "akaze")
            p1a, d1a = self._extract(im1, "akaze")

            if d0o is None and d0a is None: return np.zeros((0,2),np.float32), np.zeros((0,2),np.float32)
            if d1o is None and d1a is None: return np.zeros((0,2),np.float32), np.zeros((0,2),np.float32)

            if d0o is None: p0, d0 = p0a, d0a
            elif d0a is None: p0, d0 = p0o, d0o
            else: p0, d0 = np.vstack([p0o, p0a]), np.vstack([d0o, d0a])

            if d1o is None: p1, d1 = p1a, d1a
            elif d1a is None: p1, d1 = p1o, d1o
            else: p1, d1 = np.vstack([p1o, p1a]), np.vstack([d1o, d1a])

        if d0 is None or d1 is None or len(d0) == 0 or len(d1) == 0:
            return np.zeros((0,2),np.float32), np.zeros((0,2),np.float32)

        good = knn_ratio_and_mutual(d0, d1, self.norm, ratio=0.75)
        if not good:
            return np.zeros((0,2),np.float32), np.zeros((0,2),np.float32)

        x0 = np.array([p0[m.queryIdx] for m in good], dtype=np.float32)
        x1 = np.array([p1[m.trainIdx] for m in good], dtype=np.float32)
        return x0, x1


# ----------------- geometry / chaining -----------------
def robust_similarity(x0, x1, th=1.5, iters=5000, conf=0.999):
    if len(x0) < 3:
        return None, None
    A, inl = cv2.estimateAffinePartial2D(
        x0, x1, method=cv2.RANSAC,
        ransacReprojThreshold=th, maxIters=iters,
        confidence=conf, refineIters=30
    )
    if A is None:
        return None, None
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = A
    return H, (inl.ravel().astype(bool) if inl is not None else None)

def robust_homography(x0, x1, th=3.0, iters=5000):
    if len(x0) < 8:
        return None, None
    H, inl = cv2.findHomography(
        x0, x1, cv2.USAC_MAGSAC,
        ransacReprojThreshold=th, maxIters=iters, confidence=0.999
    )
    if H is None: return None, None
    return H, inl.ravel().astype(bool)

def chain_global(frames: List[Frame], backend: MatcherBackend,
                 model="similarity", th=3.0) -> List[np.ndarray]:
    n = len(frames)
    Ts = [np.eye(3, dtype=np.float64) for _ in range(n)]
    for j in range(1, n):
        i = j - 1
        f0, f1 = frames[i], frames[j]
        im0 = cv2.resize(f0.bgr, None, fx=f0.s_match, fy=f0.s_match, interpolation=cv2.INTER_AREA) if f0.s_match != 1.0 else f0.bgr
        im1 = cv2.resize(f1.bgr, None, fx=f1.s_match, fy=f1.s_match, interpolation=cv2.INTER_AREA) if f1.s_match != 1.0 else f1.bgr

        x0, x1 = backend.match(im0, im1)
        print(f"Frame {i}->{j}: {len(x0)} mutual-ratio matches")

        if len(x0) < (3 if model == "similarity" else 8):
            print("  [warn] not enough matches; copying previous transform")
            Ts[j] = Ts[i].copy()
            continue

        if f0.s_match != 1.0: x0 = x0 / f0.s_match
        if f1.s_match != 1.0: x1 = x1 / f1.s_match

        if model == "similarity":
            H01, inl = robust_similarity(x0, x1, th=th)
        else:
            H01, inl = robust_homography(x0, x1, th=th)

        if H01 is None:
            print("  [warn] robust model failed; copying previous transform")
            Ts[j] = Ts[i].copy()
            continue

        print(f"  > inliers: {int(inl.sum()) if inl is not None else len(x0)}")
        Ts[j] = Ts[i] @ np.linalg.inv(H01)

    return Ts


# ----------------- fusion helpers -----------------
def gaussian_pyr(img, L):
    g=[img]
    for _ in range(L-1):
        img = cv2.pyrDown(img)
        g.append(img)
    return g

def laplacian_pyr(img, L):
    gp = gaussian_pyr(img, L)
    lp = [gp[-1]]
    for i in range(L-2,-1,-1):
        up = cv2.pyrUp(gp[i+1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lp.append(cv2.subtract(gp[i], up))
    lp.reverse()
    return lp

def fuse_laplacian(a,b,wA,levels=4):
    wA = np.clip(wA,0,1).astype(np.float32)  # Fixed: .ast -> .astype
    wB = (1.0-wA).astype(np.float32)  # Fixed: .ast -> .astype
    LA, LB = laplacian_pyr(a,levels), laplacian_pyr(b,levels)
    GA = gaussian_pyr((wA*255).astype(np.uint8), levels)
    GB = gaussian_pyr((wB*255).astype(np.uint8), levels)
    out=[]
    for l in range(levels):
        wa = (GA[l].astype(np.float32)/255.0)[...,None]
        wb = (GB[l].astype(np.float32)/255.0)[...,None]
        out.append((wa*LA[l].astype(np.float32)+wb*LB[l].astype(np.float32)).astype(np.float32))
    cur = out[-1]
    for i in range(levels-2,-1,-1):
        cur = cv2.pyrUp(cur, dstsize=(out[i].shape[1], out[i].shape[0]))
        cur = cv2.add(cur, out[i])
    return np.clip(cur,0,255).astype(np.uint8)

def gradient_weight_mask(rgbA, rgbB, overlap_mask, ksize=3, eps=1e-6):
    if overlap_mask.sum()==0:
        return np.zeros_like(overlap_mask, np.float32)
    gA = cv2.Laplacian(cv2.cvtColor(rgbA, cv2.COLOR_BGR2GRAY), cv2.CV_32F, ksize=ksize)
    gB = cv2.Laplacian(cv2.cvtColor(rgbB, cv2.COLOR_BGR2GRAY), cv2.CV_32F, ksize=ksize)
    a = np.abs(gA); b = np.abs(gB); s = (a+b+eps)
    wA = np.clip(a/s,0,1)
    wA = cv2.GaussianBlur(wA,(0,0),1.5)
    wA[overlap_mask==0] = 1.0
    return wA.astype(np.float32)


# ----------------- canvas / FROI / warping -----------------
def mosaic_bounds(frames, Ts, margin=100):
    all_corners = []
    for f,T in zip(frames, Ts):
        h,w = f.bgr.shape[:2]
        corners = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]], np.float64).T
        projected = T @ corners
        projected = (projected[:2]/projected[2]).T
        all_corners.append(projected)
    all_pts = np.vstack(all_corners)
    min_xy = np.floor(all_pts.min(0)-margin).astype(int)
    max_xy = np.ceil (all_pts.max(0)+margin).astype(int)
    Wc = int(max_xy[0]-min_xy[0]); Hc = int(max_xy[1]-min_xy[1])
    offset = -min_xy
    return (Hc, Wc), offset

def dyn_froi(mask_accum, T, im_shape, offset_xy, extra=80):
    h,w = im_shape[:2]
    corners = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]], np.float64).T
    Toff = np.array([[1,0,offset_xy[0]],[0,1,offset_xy[1]],[0,0,1]], np.float64)
    H_total = Toff @ T
    proj = H_total @ corners
    proj = (proj[:2]/proj[2]).T
    x0,y0 = np.floor(proj.min(0)-extra).astype(int)
    x1,y1 = np.ceil (proj.max(0)+extra).astype(int)
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(mask_accum.shape[1], x1); y1 = min(mask_accum.shape[0], y1)
    if x1<=x0 or y1<=y0: return 0,0,0,0
    return x0,y0,x1,y1

def warp_to_patch(rgb, T, offset_xy, roi):
    x0,y0,x1,y1 = roi
    Toff = np.array([[1,0,offset_xy[0]],[0,1,offset_xy[1]],[0,0,1]], np.float64)
    H_total = Toff @ T
    T_crop = np.array([[1,0,-x0],[0,1,-y0],[0,0,1]], np.float64)
    H_patch = T_crop @ H_total
    Wp, Hp = int(x1-x0), int(y1-y0)
    warped = cv2.warpPerspective(rgb, H_patch, (Wp,Hp),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0,0,0))
    mask = cv2.warpPerspective(np.ones(rgb.shape[:2], np.uint8)*255,
                               H_patch, (Wp,Hp),
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
    return warped, mask

def auto_rotate_to_principal_axis(canvas_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) < 1000: return canvas_bgr
    pts = np.column_stack([xs, ys]).astype(np.float64)
    mean = pts.mean(0)
    cov = np.cov((pts - mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, 1]
    angle = math.degrees(math.atan2(major[1], major[0]))
    rot_deg = 90.0 - angle
    H, W = canvas_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((W/2, H/2), rot_deg, 1.0)
    corners = np.array([[0,0,1],[W,0,1],[W,H,1],[0,H,1]], dtype=np.float32).T
    A = np.vstack([M, [0,0,1]]).astype(np.float32)
    out = A @ corners
    xs2, ys2 = out[0], out[1]
    minx, miny = xs2.min(), ys2.min()
    maxx, maxy = xs2.max(), ys2.max()
    Wn = int(np.ceil(maxx - minx)); Hn = int(np.ceil(maxy - miny))
    M[0,2] -= minx; M[1,2] -= miny
    return cv2.warpAffine(canvas_bgr, M, (Wn, Hn), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))


# ----------------- COMPOSITOR with coverage governor -----------------
def compose_incremental(frames, Ts, out_png,
                        lap_levels=4, ram_guard_gb=6.0,
                        tile_margin=0, auto_rotate=False,
                        max_coverage=3, sharper_override=False):
    (Hc,Wc), offset = mosaic_bounds(frames, Ts)
    est_gb = (Hc * Wc * 4) / 1e9
    if est_gb > ram_guard_gb:
        scale = math.sqrt(ram_guard_gb / est_gb)
        print(f"[RAM guard] Canvas too large ({Hc}x{Wc}px, ~{est_gb:.1f}GB). Downscaling by x{scale:.2f}")
        S = np.array([[scale,0,0],[0,scale,0],[0,0,1]], dtype=np.float64)
        Ts = [S @ T @ np.linalg.inv(S) for T in Ts]
        (Hc,Wc), offset = mosaic_bounds(frames, Ts)

    print(f"Creating canvas of size {Wc}x{Hc}px")
    canvas = np.zeros((Hc,Wc,3), np.uint8)
    mask_accum = np.zeros((Hc,Wc), np.uint8)
    coverage_count = np.zeros((Hc,Wc), np.uint8)  # how many images wrote each pixel

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".partial.png")

    for idx, (f, T) in enumerate(zip(frames, Ts)):
        x0,y0,x1,y1 = dyn_froi(mask_accum, T, f.bgr.shape, offset, extra=80)
        if x1<=x0 or y1<=y0: 
            continue

        x0=max(0,x0-tile_margin); y0=max(0,y0-tile_margin)
        x1=min(Wc,x1+tile_margin); y1=min(Hc,y1+tile_margin)
        roi = (x0,y0,x1,y1)

        warped, wmask = warp_to_patch(f.bgr, T, offset, roi)
        sub = canvas[y0:y1, x0:x1]
        submask = mask_accum[y0:y1, x0:x1]
        subcov = coverage_count[y0:y1, x0:x1]

        # Regions
        new_only = (wmask>0) & (submask==0)
        overlap  = (wmask>0) & (submask>0)

        # ---- Coverage gating ----
        allow_new_only  = new_only  & (subcov < max_coverage)
        allow_overlap   = overlap   & (subcov < max_coverage)

        rejected_new = int(new_only.sum() - allow_new_only.sum())
        rejected_ovr = int(overlap.sum()  - allow_overlap.sum())

        # 1) write new-only where allowed
        if allow_new_only.any():
            sub[allow_new_only] = warped[allow_new_only]
            submask[allow_new_only] = 255
            subcov[allow_new_only] = np.minimum(subcov[allow_new_only] + 1, 255)

        # 2) blend overlaps where allowed
        if allow_overlap.any():
            wA = gradient_weight_mask(sub, warped, allow_overlap)
            blended = fuse_laplacian(sub, warped, wA, levels=lap_levels)
            sub[allow_overlap] = blended[allow_overlap]
            subcov[allow_overlap] = np.minimum(subcov[allow_overlap] + 1, 255)

        # 3) Optional sharper-override: replace capped pixels only if new is sharper
        if sharper_override:
            capped_zone = overlap & (subcov >= max_coverage)
            if capped_zone.any():
                gray_sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
                gray_w   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                # Sobel magnitude (cheap)
                gx1 = cv2.Sobel(gray_sub, cv2.CV_32F, 1, 0, ksize=3)
                gy1 = cv2.Sobel(gray_sub, cv2.CV_32F, 0, 1, ksize=3)
                gx2 = cv2.Sobel(gray_w,   cv2.CV_32F, 1, 0, ksize=3)
                gy2 = cv2.Sobel(gray_w,   cv2.CV_32F, 0, 1, ksize=3)
                mag1 = cv2.magnitude(gx1, gy1)
                mag2 = cv2.magnitude(gx2, gy2)
                # Require new to be noticeably sharper (10% more)
                better = capped_zone & (mag2 > 1.10 * (mag1 + 1e-3))
                if better.any():
                    sub[better] = warped[better]
                    # note: do NOT increase subcov (swap, not add)

        # commit ROI
        canvas[y0:y1, x0:x1] = sub
        mask_accum[y0:y1, x0:x1] = submask
        coverage_count[y0:y1, x0:x1] = subcov

        # Progress + small stats
        if (idx+1)%2==0 or (idx+1)==len(frames):
            pct_rej = 100.0 * (rejected_new + rejected_ovr) / max(1, (new_only.sum()+overlap.sum()))
            print(f"  frame {idx+1}/{len(frames)}: rejected for cap ~{pct_rej:.1f}%")
        if (idx+1)%5==0 or (idx+1) == len(frames):
            Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(tmp_path)
            print(f"  saved checkpoint → {tmp_path.name}")

    if auto_rotate:
        print("Auto-rotating mosaic to vertical principal axis...")
        canvas = auto_rotate_to_principal_axis(canvas, mask_accum>0)

    print("Finalizing mosaic...")
    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(out_png)
    try:
        if tmp_path.exists(): tmp_path.unlink()
    except: pass
    return str(out_png)


# ----------------- CLI -----------------
def main():
    import argparse
    ap = argparse.ArgumentParser("UAV nadir fast-stitch (ORB/AKAZE) with overlap cap")
    ap.add_argument("--folder", help="Image folder (UAV photo set)")
    ap.add_argument("--out", default="mosaic.png", help="Output PNG file")

    ap.add_argument("--matcher", default="orb", choices=["orb", "akaze", "hybrid"],
                    help="Feature matcher (hybrid=ORB+AKAZE union)")
    ap.add_argument("--match-max-side", type=int, default=2000,
                    help="Resize images for matching only")
    ap.add_argument("--max-features", type=int, default=5000,
                    help="Max ORB features (if matcher=orb/hybrid)")
    ap.add_argument("--grid-x", type=int, default=10, help="Grid columns for uniform KP sampling")
    ap.add_argument("--grid-y", type=int, default=8,  help="Grid rows for uniform KP sampling")
    ap.add_argument("--per-cell", type=int, default=300, help="Max keypoints per grid cell")

    ap.add_argument("--model", default="similarity", choices=["similarity","homography"],
                    help="Geometric model. 'similarity' enforces nadir-like ortho view")

    ap.add_argument("--lap-levels", type=int, default=4, help="Laplacian pyramid levels")
    ap.add_argument("--ram-guard-gb", type=float, default=6.0,
                    help="Attempt to keep canvas memory under this limit (GB)")
    ap.add_argument("--auto-rotate", action="store_true",
                    help="Rotate final mosaic so dominant axis is vertical")

    # overlap governor
    ap.add_argument("--max-coverage", type=int, default=3,
                    help="Max number of images allowed to contribute to the same pixel")
    ap.add_argument("--sharper-override", action="store_true",
                    help="If capped, allow replace only when new pixel is sharper")

    args = ap.parse_args()

    if not args.folder:
        try:
            folder_path = input("Enter photo folder: ").strip()
            args.folder = folder_path.strip('"')
        except EOFError:
            print("\nNo folder provided. Exiting.")
            sys.exit(1)

    out_png = args.out if str(args.out).lower().endswith(".png") else str(Path(args.out).with_suffix(".png"))
    print(f"Output mosaic PNG: {out_png}")

    img_dir = Path(args.folder)
    if not img_dir.exists():
        print(f"Folder not found: {img_dir}")
        sys.exit(1)

    frames = load_frames(img_dir, match_max_side=args.match_max_side)
    if len(frames)<2:
        print("Need at least 2 images.")
        sys.exit(1)

    backend = MatcherBackend(
        matcher_type=args.matcher, max_features=args.max_features,
        grid=(args.grid_x, args.grid_y), per_cell=args.per_cell
    )

    print("\n== Building global pairwise transforms (sequential) ==")
    Ts = chain_global(frames, backend, model=args.model, th=3.0)

    print("\n== Warping & seam-aware fusion with overlap cap ==")
    out_path = compose_incremental(
        frames, Ts, out_png=out_png,
        lap_levels=args.lap_levels,
        ram_guard_gb=args.ram_guard_gb,
        auto_rotate=args.auto_rotate,
        max_coverage=args.max_coverage,
        sharper_override=args.sharper_override
    )
    print(f"\nDone. Saved mosaic to {out_path}")

if __name__=="__main__":
    main()
