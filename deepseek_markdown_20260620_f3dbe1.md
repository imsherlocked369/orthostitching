<!-- 
  Math rendering note:
  To view equations correctly, open this file in an environment that supports MathJax, e.g.:
    - Typora (built-in)
    - VS Code with "Markdown+Math" extension
    - Jupyter notebook
    - GitHub (simply preview the file)
    - Or convert to HTML/PDF using Pandoc: 
      pandoc report.md --katex -o report.html
-->

# Report: Mathematical Foundations of a UAV Nadir Fast-Stitch Pipeline

**Focus:** Feature-based alignment, geometric modelling, multi-image fusion, and overlap control for ortho-mosaic generation.

---

## 1. Introduction
The provided script implements a computationally efficient UAV image stitching pipeline without neural networks. It relies on hand-crafted keypoint detectors, geometric alignment under a nadir assumption, multi-scale blending, and adaptive overlap control. The output is a rectified orthomosaic suitable for geospatial analysis and mapping applications.

---

## 2. Feature Detection and Description

### 2.1 Detectors: ORB and AKAZE
The pipeline supports **ORB** (Oriented FAST and Rotated BRIEF) [1] and **AKAZE** (Accelerated-KAZE) [2], optionally combined in a "hybrid" mode.
- **ORB** uses a FAST-9 corner detector, Harris corner measure to select the best corners, and a steered BRIEF descriptor. The keypoint orientation is computed from intensity centroid moments:
  
  $$
  m_{pq} = \sum_{x,y} x^p y^q I(x,y), \quad
  \theta = \arctan2(m_{01}, m_{10})
  $$
  
- **AKAZE** builds a nonlinear scale space via Fast Explicit Diffusion (FED), then detects features from Hessian responses and computes a Modified-Local Difference Binary (M-LDB) descriptor.

### 2.2 Grid-Uniform Keypoint Distribution
To prevent clusters of features in highly textured regions, the code partitions each image into a regular grid of $g_x \times g_y$ cells and retains only the top $N_{\text{per\_cell}}$ keypoints (by response magnitude). This ensures uniform spatial coverage and reduces redundancy.

---

## 3. Feature Matching

### 3.1 KNN Ratio Test with Mutual Consistency
Correspondences are obtained using brute-force matching with a **ratio test** [4]:

For a query descriptor $d_q$ and its two nearest neighbours $d_{n1}, d_{n2}$, a match is accepted if

$$
\frac{\|d_q - d_{n1}\|}{\|d_q - d_{n2}\|} < \tau ,
$$

where $\tau = 0.75$ (Hamming distance for ORB/AKAZE binary descriptors). Mutual consistency (cross-check) is enforced: a match is kept only if $q \rightarrow n$ and $n \rightarrow q$ are both the best matches in their respective directions. This reduces false positives.

### 3.2 Scale Handling
Images are resized for matching (controlled by `match_max_side`) to reduce computational load, but keypoint coordinates are later scaled back to original resolution before geometric fitting.

---

## 4. Geometric Alignment

### 4.1 Pairwise Transformation Models
Two models are available:
- **Similarity (partial affine 2D):** $T \in \mathbb{R}^{2\times 3}$ with uniform scale, rotation, and translation. Estimated via `cv2.estimateAffinePartial2D` which minimises reprojection error:

  $$
  \min_{s,R,\mathbf{t}} \sum_i \left\| \mathbf{x}_i' - (sR \mathbf{x}_i + \mathbf{t}) \right\|^2
  $$

  with RANSAC [5] to reject outliers. This model enforces a nadir-like ortho-rectified view.
  
- **Homography:** $H \in \mathbb{R}^{3\times 3}$ estimated with USAC-MAGSAC [6], a robust variant that marginalises over noise scale.

### 4.2 Global Alignment by Sequential Chaining
With $n$ images, the transform $M_j$ mapping image $j$ to the global canvas is obtained by concatenating pairwise inverses:

$$
M_0 = I_3,\qquad
M_j = M_i \circ H_{ij}^{-1} \quad (i=j-1, j=1\dots n-1)
$$

where $H_{ij}$ is the map from image $i$ to $j$. This assumes a sequential flight path; bundle adjustment is omitted for speed.

---

## 5. Image Warping and Canvas Construction

### 5.1 Canvas Bounds and Offset
All image corners are projected using their global transforms. The axis-aligned bounding box of the projected points, enlarged by a margin, defines the canvas dimensions $(W_c, H_c)$. An offset vector is computed to ensure all coordinates are non-negative:

$$
\mathbf{o} = -\min(\text{projected points}) + \text{margin}.
$$

### 5.2 Dynamic FROI (Feature-based Region of Interest)
For each image, a tight rectangular ROI is computed from its projected bounding box extended by a small extra border. This avoids warping the entire canvas, drastically reducing memory usage.

### 5.3 Warping
A pixel at location $\mathbf{p}$ on the canvas corresponds to the source coordinate $\mathbf{p}_{\text{src}}$ via:

$$
\tilde{\mathbf{p}}_{\text{src}} = T_{\text{crop}} \cdot T_{\text{off}} \cdot M_j \; \tilde{\mathbf{p}} ,
$$

where $T_{\text{off}}$ incorporates the offset $\mathbf{o}$, and $T_{\text{crop}}$ shifts the ROI origin to $(0,0)$. Actual sampling uses inverse mapping (bilinear interpolation) with `cv2.warpPerspective`.

---

## 6. Blending and Compositing

### 6.1 Laplacian Pyramid Blending
To smooth seams, a multi-scale fusion from [7] is employed. An image $I$ is decomposed into a Gaussian pyramid $\{G_0 = I, G_1, \dots, G_{L-1}\}$ and a Laplacian pyramid $\{L_0, \dots, L_{L-1}\}$ where each Laplacian level is:

$$
L_k = G_k - \text{expand}(G_{k+1}), \quad \text{expand}(\cdot) = \text{pyrUp}(\cdot).
$$

A weighting mask $W_A$ (and $W_B=1-W_A$) is similarly built into a Gaussian pyramid. The blended pyramid is formed by

$$
O_k = W_A^{(k)} \cdot L_A^{(k)} + W_B^{(k)} \cdot L_B^{(k)},
$$

and the final blended image is reconstructed by successive upsample-and-add operations.

### 6.2 Gradient-Based Weight Mask
Instead of a simple distance transform, the weight mask for overlapping regions is derived from local gradient magnitudes [8]:

$$
W_A(\mathbf{p}) = \frac{\|\nabla I_A(\mathbf{p})\|}{\|\nabla I_A(\mathbf{p})\| + \|\nabla I_B(\mathbf{p})\| + \epsilon},
$$

where $\nabla$ is approximated by a Laplacian or Sobel operator. This gives higher influence to sharper details, preserving texture.

---

## 7. Overlap Governor and Sharpness Override

### 7.1 Coverage Cap
A per-pixel counter `coverage_count` tracks how many images have already contributed. New pixels are written only if the count is below a threshold $C_{\max}$ (default 3). This prevents over-blending and reduces computational overhead.

### 7.2 Sharper-Override Option
Even when a pixel has reached $C_{\max}$, it may be replaced if the new image provides a **significantly sharper** local patch. Sharpness is measured by the gradient magnitude:

$$
S(\mathbf{p}) = \sqrt{ (\partial_x I)^2 + (\partial_y I)^2 }.
$$

A pixel is overwritten if

$$
S_{\text{new}}(\mathbf{p}) > (1+\alpha) \cdot S_{\text{existing}}(\mathbf{p}) ,
$$

with $\alpha = 0.10$. This effectively performs a local maximum contrast selection, similar to focus-stacking methods [9].

---

## 8. Post-processing: Auto-Rotate

The dominant orientation of the mosaic is computed via PCA of foreground pixel coordinates (mask > 0). The principal axis (largest eigenvector of the covariance matrix) is aligned with the vertical, yielding a rotation angle:

$$
\theta = 90^\circ - \arctan2(\text{eigvec}_y, \text{eigvec}_x).
$$

This simple rectification assumes the scene's long edge is roughly vertical, often true for flight-strip mosaics.

---

## 9. Conclusion
The pipeline integrates well-established computer vision techniques—binary features, robust pairwise estimation, multi-scale blending, and overlap management—to produce UAV ortho-mosaics efficiently without deep learning. The sequential chaining assumption trades global optimality for computational speed, making it suitable for real-time or near-real-time UAV workflows.

---

## References

1. Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. *ICCV*.
2. Alcantarilla, P. F., Nuevo, J., & Bartoli, A. (2013). Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces. *BMVC*.
3. Mur-Artal, R., Montiel, J. M. M., & Tardós, J. D. (2015). ORB-SLAM: a versatile and accurate monocular SLAM system. *IEEE TRO*.
4. Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *IJCV*.
5. Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. *Comm. ACM*.
6. Barath, D., Matas, J., & Noskova, J. (2020). MAGSAC: marginalizing sample consensus. *IEEE TPAMI*.
7. Burt, P. J., & Adelson, E. H. (1983). The Laplacian pyramid as a compact image code. *IEEE Trans. Comm.*
8. Agarwala, A., Dontcheva, M., Agrawala, M., Drucker, S., Colburn, A., Curless, B., Salesin, D., & Cohen, M. (2004). Interactive digital photomontage. *ACM TOG (SIGGRAPH)*. (Gradient-domain blending).
9. Goshtasby, A. A. (2005). Fusion of multi-exposure images. *Image and Vision Computing*.

---

*Report generated from the provided Python implementation (v2026-06).*
