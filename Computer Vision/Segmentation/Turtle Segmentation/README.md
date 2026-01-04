# Turtle Segmentation → Tight Polygon Extraction (Colab)

This Colab notebook post-processes a turtle **segmentation mask** to compute a **tight enclosing polygon** by extracting boundary pixels and constructing a **convex hull** (Graham scan), then overlays the polygon on the source image.

---

## What it does (1 sentence)
Converts a binary turtle segmentation mask into an ordered polygon (convex hull) and visualizes the polygon overlay on the original image.

---

## Why it matters (use-case)
Segmentation masks are useful, but many downstream tasks need a compact geometric representation:
- exporting annotations as polygons (COCO-style)
- tracking and shape/area/perimeter measurements
- computing tight ROI crops for classification
- deployment where polygons are cheaper than full-resolution masks

---

## Results (metrics, latency, size, FPS, accuracy)
**Output:** `torch.Tensor` of polygon vertices in `(y, x)` format.

**Complexity**
- Boundary extraction: **O(H·W)** (scan mask once + constant neighbor checks)
- Convex hull: **O(B log B)** where **B** is number of boundary pixels
- Memory: **O(B)** for boundary point storage

> Tip: Add a `%%timeit` cell in Colab to report `ms/mask` for your typical input resolution.

---

## Approach (diagram or bullets)

**Pipeline**
1. Input: `test_mask` where `1 = turtle`, `0 = background`
2. Extract boundary points: a foreground pixel is a boundary if any of its 4-neighbors is background
3. Choose pivot point (lowest-then-leftmost)
4. Sort boundary points by polar angle around pivot (`atan2`)
5. Run **Graham scan** with a stack using cross-product orientation checks
6. Return polygon vertices + overlay on image

**Mental model**
`mask → boundary pixels → angle sort → hull stack → polygon vertices → plot overlay`

---

## Run it (Colab)
1. Open the notebook in Google Colab.
2. Run all cells top-to-bottom.
3. The final cells will:
   - compute `polygon_points_n2_tensor = get_tight_polygon_from_mask(test_mask_tensor)`
   - visualize with `visualize_polygon_on_image(test_image_tensor, polygon_points_n2_tensor)`

---

## Usage (end result)

```python
polygon_points_n2_tensor = get_tight_polygon_from_mask(test_mask_tensor)
visualize_polygon_on_image(test_image_tensor, polygon_points_n2_tensor)
