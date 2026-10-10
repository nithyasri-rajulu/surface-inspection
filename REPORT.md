# Golden Reference Surface Defect Inspection Project Report

## Purpose and provenance

An independent quality-inspection demonstrator for one textured, planar part viewed under a fixed camera. The implementation was prepared in October 2026 for a mechatronics and robotics portfolio.

## Engineering approach

Translation-only ECC registration aligns each candidate to a golden reference. A robust linear brightness fit reduces global lighting differences. Absolute residuals are thresholded, cleaned with morphological operations and grouped into connected defect regions. Low registration confidence causes an explicit review/error result instead of silently passing an image. Defect masks are represented in the aligned reference coordinate system.

## Reproduction

Install the pinned development requirements in a clean Python 3.12 environment. Run `python -m pytest -q`, then `python app.py --demo`. Configuration, fixed seeds and ground-truth generation are visible in `app.py`. Actual outputs are recorded in `outputs/results.json`.

## Evaluation

4 tests passed. On 40 generated images (20 defective, 20 clean), the fixed pipeline produced 20 true positives, 20 true negatives, no false positives, no false negatives and no registration failures. Mean defect-pixel IoU was 0.919. This is a synthetic benchmark with simple scratches/spots, translations up to 3 pixels, global brightness changes and noise.

![Pipeline output](outputs/preview.png)

## Interpretation and limitations

The alignment model only covers translation, not rotation, perspective or deformation. A golden image must represent normal part variation. Reflective texture, shadows, drift and unseen defects can increase false alarms. The synthetic benchmark has no real production images and no learned defect model; its perfect image-level classification is not a factory performance estimate. The current evidence supports this limited demonstration. Independent tests on new real inputs are needed before presenting broader performance claims.

## Next experiment

Collect normal and defective samples from one real part family; define separate development and held-out test sets. Measure precision, recall, per-defect sensitivity and the review rate. Add pose registration and compare an anomaly-detection method when enough images are available.

## How to explain this in an interview

Explain the input assumptions first, then walk through the pipeline and a failure case. Show the test and generated result that supports each numerical claim. Be able to change one threshold, rerun the experiment and explain how it affects the result.

## References

- https://docs.opencv.org/4.x/dc/d6b/group__video__track.html
- https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
