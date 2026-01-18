# UAV Fast-Stitching (ORB/AKAZE) — OpenCV-Only Mosaic Builder

Fast, practical UAV  photo stitching (nadir)  using OpenCV ORB/AKAZE only (no neural backends). Designed for sequential drone captures where a similarity transform (scale/rotation/translation) is often a better fit than a full homography.

This version adds a pixel-wise overlap governor (coverage cap) and an optional sharper override that can replace capped pixels only when the new view is measurably sharper.



Features

- OpenCV-only stitching: ORB / AKAZE / Hybrid (union of both)
- Grid-uniform keypoint sampling for better spatial coverage
- Mutual + ratio test matching (robust against bad correspondences)
- Robust geometric estimation
  - `similarity` (via `estimateAffinePartial2D` → 3×3 similarity-like warp)
  - `homography` (via `findHomography` with `USAC_MAGSAC`)
- Adaptive FROI (dynamic frame region-of-interest on canvas) to avoid full-canvas warps
- Seam-aware fusion
  - gradient-based overlap weights
  - Laplacian pyramid blending
- Pixel-wise overlap governor
  - cap maximum number of contributing images per pixel (`--max-coverage`, default 3)
  - optional sharper override on capped overlap pixels (`--sharper-override`)
- **RAM guard**: downscales transforms/canvas if estimated memory exceeds `--ram-guard-gb`


## Requirements

- Python 3.9+
- opencv-python
- numpy
- Pillow

