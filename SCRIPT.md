# Video Presentation Script (3 minutes)
# Sample4Geo XAI Analysis — BBM416 Computer Vision

Notes:
- Total duration: ~3 minutes (180 seconds)
- Screen-share the notebook in Colab
- Scroll to the relevant cell for each section, show outputs/figures
- Follow the timing targets below

---

## [0:00–0:20] Introduction (Cell 0 — Intro markdown)

**Show:** The title cell of the notebook (Cell 0)

**Say:**

"Hi, we are Kuzey Ersoy and Halis Yucel. For our BBM416 Computer Vision
project, we performed an Explainable AI analysis on Sample4Geo, a cross-view
geo-localization model published at ICCV 2023. The model takes a drone or
street-level photo and retrieves the matching satellite image from a database.
We used two XAI methods to understand what the model actually looks at when
making these decisions: GradCAM and Occlusion Sensitivity."

---

## [0:20–0:45] Methods (Cell 19, 22, 24 — XAI Setup)

**Show:** The method table in Cell 19, then briefly scroll through Cell 22 (GradCAM class) and Cell 24 (Occlusion class)

**Say:**

"We chose two methods that work on fundamentally different principles.
GradCAM is gradient-based and white-box: it backpropagates the cosine
similarity score through the last convolutional layer to produce a smooth
heatmap. Occlusion Sensitivity is perturbation-based and black-box: it
slides a 64-by-64 gray patch across the image and measures how much the
similarity drops at each position. For both methods, we also compute
faithfulness curves by progressively masking the most important pixels
and tracking the score degradation."

---

## [0:45–1:05] Baseline Results (Cell 14, 17, 18 output)

**Show:** Cell 14 output (U1652 metrics), Cell 17 output (VIGOR same), Cell 18 output (VIGOR cross)

**Say:**

"First we verified the model's baseline performance. On University-1652,
Recall at 1 is 92.66 percent. On VIGOR same-area it's 77.86, and cross-area
61.71. These match the published results. University-1652 is a drone-to-satellite
task where both views are roughly top-down, so it's easier. VIGOR matches
street panoramas to satellite tiles, which is a much harder problem due to
the extreme viewpoint difference."

---

## [1:05–1:35] GradCAM Results (Cell 34, 35, 45, 46 figures)

**Show:** One figure from Cell 34 (U1652 successful), one from Cell 35 (U1652 failed), then Cell 46 (VIGOR failed). Point at heatmaps and faithfulness curves.

**Say:**

"Looking at GradCAM results on University-1652: for successful matches,
the model focuses on building edges and rooftop transitions in the drone
view, and the same building footprint in the satellite view. The faithfulness
curve drops monotonically, confirming correct feature identification.

For failed matches, the query still attends to a meaningful structure, but
the wrong gallery image shows scattered, diffuse activation. The model
cannot find a discriminative anchor in the incorrect image.

On VIGOR, the most interesting failure case is a street view taken under
an elevated railway. The model fixates on the steel structure overhead,
which is completely invisible from satellite. The faithfulness curve is
V-shaped: masking certain pixels actually increases the similarity score,
proving these features are actively misleading the model."

---

## [1:35–2:05] Occlusion Sensitivity Results (Cell 56 — Display Saved Results, occlusion figures)

**Show:** Occlusion figures: U1652 successful first, then U1652 failed (point at the faithfulness spike), then VIGOR failed.

**Say:**

"Occlusion Sensitivity highlights the same regions as GradCAM, which
cross-validates both methods. The heatmaps are blockier because we use
a 64-pixel patch with stride 32, but the spatial localization is consistent.

The key finding is in the faithfulness curves. For successful matches, the
curve drops smoothly as we mask important pixels. But for failed matches,
the gallery curve spikes around 80 percent masking. This means removing the
pixels the model focused on temporarily improves the score. This is
mathematical proof that the model latched onto misleading features that
were inflating similarity to the wrong target."

---

## [2:05–2:35] Comparison and Analysis (Cell 51 — Analysis markdown)

**Show:** Cell 51 analysis markdown or the comparison table from the report

**Say:**

"Comparing the two methods: GradCAM is fast and produces smooth heatmaps
but requires access to model internals. Occlusion Sensitivity is
model-agnostic and provides direct causal evidence but is computationally
heavier and produces coarser maps.

We identified two distinct failure modes. First, diffuse gallery attention
in University-1652: the wrong satellite image has no matching structure, so
activation spreads everywhere. Second, cross-view invisible features in VIGOR:
the model attends to structures only visible from street level, like overhead
rail infrastructure, that have no counterpart in satellite imagery."

---

## [2:35–3:00] Conclusion (Cell 52 output — results summary)

**Show:** Cell 52 output (results summary dictionary)

**Say:**

"In conclusion, both XAI methods consistently show that the model relies
on building geometry and structural boundaries for successful matching.
Failures are interpretable and follow two clear patterns. For future work,
incorporating depth estimation or multi-scale attention could help the model
avoid fixating on view-dependent features.

All code, weights, and results are publicly available on our GitHub repository
and Google Drive. Thank you for watching."

---

## Timing Summary

| Section | Duration | Cumulative |
|---------|----------|------------|
| Introduction | 20s | 0:20 |
| Methods | 25s | 0:45 |
| Baseline | 20s | 1:05 |
| GradCAM | 30s | 1:35 |
| Occlusion | 30s | 2:05 |
| Comparison | 30s | 2:35 |
| Conclusion | 25s | 3:00 |

## Notebook Cell Reference

| Presentation Section | Cell(s) to Show |
|---|---|
| Introduction | Cell 0 (markdown) |
| Methods | Cell 19 (table), Cell 22 (GradCAM code), Cell 24 (Occlusion code) |
| Baseline | Cell 14 output, Cell 17 output, Cell 18 output |
| GradCAM U1652 | Cell 34 output (successful), Cell 35 output (failed) |
| GradCAM VIGOR | Cell 45 output (successful), Cell 46 output (failed) |
| Occlusion | Cell 37/38 output or Cell 56 (Display Saved Results, occlusion section) |
| Analysis | Cell 51 (markdown), Cell 52 output (summary) |
