# Multi-face analysis summary (auto-generated)

Target faces with results: 2816 of 2818 | images: 1250 (mixed: 428) | chunks present: [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5)]... (32 of 32) | bootstrap 2000 resamples of images

## Accuracy per arm (95% CI)
- **all / acc_all**: Plain (position) 0.675 [0.655, 0.695] (n=2816); Box (position) 0.671 [0.651, 0.692] (n=2816); Box+Number (number) 0.726 [0.707, 0.745] (n=2816); SoV (number) 0.527 [0.505, 0.549] (n=2816); Box+Number (position) 0.599 [0.567, 0.628] (n=997); SoV (position) 0.475 [0.444, 0.506] (n=997)
- **mixed / acc_all**: Plain (position) 0.533 [0.502, 0.563] (n=997); Box (position) 0.531 [0.499, 0.561] (n=997); Box+Number (number) 0.608 [0.575, 0.637] (n=997); SoV (number) 0.480 [0.450, 0.510] (n=997); Box+Number (position) 0.599 [0.567, 0.628] (n=997); SoV (position) 0.475 [0.444, 0.506] (n=997)
- **all / acc_expr**: Plain (position) 0.256 [0.194, 0.318] (n=301); Box (position) 0.249 [0.190, 0.310] (n=301); Box+Number (number) 0.266 [0.209, 0.325] (n=301); SoV (number) 0.123 [0.085, 0.166] (n=301); Box+Number (position) 0.222 [0.164, 0.284] (n=225); SoV (position) 0.147 [0.097, 0.198] (n=225)
- **mixed / xref**: Plain (position) 0.313 [0.285, 0.341] (n=997); Box (position) 0.298 [0.271, 0.327] (n=997); Box+Number (number) 0.241 [0.214, 0.269] (n=997); SoV (number) 0.331 [0.302, 0.361] (n=997); Box+Number (position) 0.244 [0.217, 0.271] (n=997); SoV (position) 0.318 [0.290, 0.349] (n=997)

## Paired differences, points (95% CI)
- **all / acc_all**
    - SoV as designed vs Plain (sov|numbered - plain|positional): -14.9 [-17.0, -12.7]  *
    - Box+Number as designed vs Plain (box_number|numbered - plain|positional): +5.0 [+3.4, +6.6]  *
    - Boxes alone vs Plain (same prompt) (box|positional - plain|positional): -0.4 [-1.7, +0.9]
    - Numbers + numbered prompt vs boxes (box_number|numbered - box|positional): +5.5 [+4.0, +7.0]  *
    - Landmarks (same overlay type, same prompt) (sov|numbered - box_number|numbered): -19.9 [-22.0, -18.0]  *
    - Using the number vs counting (same image) (box_number|numbered - box_number|positional): +0.9 [-0.4, +2.2]
    - Using the number vs counting (SoV image) (sov|numbered - sov|positional): +0.5 [-0.9, +1.8]
    - Number tags themselves (same prompt) (box_number|positional - box|positional): +6.8 [+4.7, +9.1]  *
    - SoV marks only (same prompt) vs Plain (sov|positional - plain|positional): -5.7 [-8.8, -2.6]  *
- **mixed / acc_all**
    - SoV as designed vs Plain (sov|numbered - plain|positional): -5.2 [-8.3, -2.3]  *
    - Box+Number as designed vs Plain (box_number|numbered - plain|positional): +7.5 [+4.7, +10.3]  *
    - Boxes alone vs Plain (same prompt) (box|positional - plain|positional): -0.2 [-2.2, +1.8]
    - Numbers + numbered prompt vs boxes (box_number|numbered - box|positional): +7.7 [+5.2, +10.2]  *
    - Landmarks (same overlay type, same prompt) (sov|numbered - box_number|numbered): -12.7 [-15.8, -9.9]  *
    - Using the number vs counting (same image) (box_number|numbered - box_number|positional): +0.9 [-0.4, +2.2]
    - Using the number vs counting (SoV image) (sov|numbered - sov|positional): +0.5 [-0.9, +1.8]
    - Number tags themselves (same prompt) (box_number|positional - box|positional): +6.8 [+4.7, +9.1]  *
    - SoV marks only (same prompt) vs Plain (sov|positional - plain|positional): -5.7 [-8.8, -2.6]  *
- **mixed / xref**
    - SoV as designed vs Plain (sov|numbered - plain|positional): +1.8 [-0.8, +4.5]
    - Box+Number as designed vs Plain (box_number|numbered - plain|positional): -7.2 [-9.6, -4.7]  *
    - Boxes alone vs Plain (same prompt) (box|positional - plain|positional): -1.5 [-3.3, +0.4]
    - Numbers + numbered prompt vs boxes (box_number|numbered - box|positional): -5.7 [-7.9, -3.5]  *
    - Landmarks (same overlay type, same prompt) (sov|numbered - box_number|numbered): +9.0 [+6.5, +11.7]  *
    - Using the number vs counting (same image) (box_number|numbered - box_number|positional): -0.3 [-1.6, +0.9]
    - Using the number vs counting (SoV image) (sov|numbered - sov|positional): +1.3 [+0.1, +2.5]  *
    - Number tags themselves (same prompt) (box_number|positional - box|positional): -5.4 [-7.4, -3.4]  *
    - SoV marks only (same prompt) vs Plain (sov|positional - plain|positional): +0.5 [-2.3, +3.3]
- **all / acc_expr**
    - SoV as designed vs Plain (sov|numbered - plain|positional): -13.3 [-19.3, -7.5]  *
    - Box+Number as designed vs Plain (box_number|numbered - plain|positional): +1.0 [-3.8, +5.6]
    - Boxes alone vs Plain (same prompt) (box|positional - plain|positional): -0.7 [-4.0, +2.8]
    - Numbers + numbered prompt vs boxes (box_number|numbered - box|positional): +1.7 [-2.6, +5.8]
    - Landmarks (same overlay type, same prompt) (sov|numbered - box_number|numbered): -14.3 [-20.2, -8.6]  *
    - Using the number vs counting (same image) (box_number|numbered - box_number|positional): +0.9 [-1.5, +3.3]
    - Using the number vs counting (SoV image) (sov|numbered - sov|positional): -3.1 [-7.3, +0.5]
    - Number tags themselves (same prompt) (box_number|positional - box|positional): +2.2 [-1.7, +6.0]
    - SoV marks only (same prompt) vs Plain (sov|positional - plain|positional): -4.9 [-11.4, +1.8]
- **not_mixed / acc_all**
    - SoV as designed vs Plain (sov|numbered - plain|positional): -20.2 [-23.0, -17.4]  *
    - Box+Number as designed vs Plain (box_number|numbered - plain|positional): +3.7 [+1.7, +5.7]  *
    - Boxes alone vs Plain (same prompt) (box|positional - plain|positional): -0.5 [-2.1, +1.0]
    - Numbers + numbered prompt vs boxes (box_number|numbered - box|positional): +4.2 [+2.5, +6.0]  *
    - Landmarks (same overlay type, same prompt) (sov|numbered - box_number|numbered): -23.9 [-26.4, -21.2]  *

## Does the SoV-vs-Plain effect differ between groups? (acc_all, 'SoV as designed vs Plain'; * = CI excludes 0)
- race_group: Asian vs Caucasian-like: +0.9 [-4.2, +6.0]
- race_group: Black vs Caucasian-like: +2.4 [-3.6, +9.0]
- gender: Male vs Female: +0.8 [-3.0, +4.7]
- age_band: 0-19 vs 20-29: +0.6 [-5.5, +6.6]
- age_band: 30-39 vs 20-29: -0.6 [-7.3, +5.6]
- age_band: 40-59 vs 20-29: -6.8 [-13.5, -0.6]  *
- age_band: 60+ vs 20-29: -8.1 [-16.8, +0.6]
- face_size: medium vs small: +14.0 [+8.4, +19.2]  *
- face_size: large vs small: +25.7 [+20.7, +30.4]  *
- image_type: mixed vs not_mixed: +15.0 [+10.9, +19.1]  *
- reference_clear: extra/missing detections vs detections = targets: -2.3 [-6.7, +2.4]
- n_targets: 3+ faces vs 2 faces: -18.3 [-23.6, -13.0]  *

## Warnings
- Emotions are very unbalanced (mostly Happy and Neutral; Neutral is never predicted). acc_all mostly measures Happy; use acc_expr and xref for the informative faces.
- Group labels are FairFace ESTIMATES. Beyond Happy, the group cells are too small for group claims; cells below 30 faces: Asian/angry=5, Asian/disgust=13, Asian/fear=0, Asian/sad=20, Asian/surprise=17, Black/angry=0, Black/disgust=11, Black/fear=0, Black/sad=4, Black/surprise=7, Caucasian-like/fear=9
- Positional prompts make plain/box rely on the model counting faces; 'reference_clear' rows compare images where the number of detected faces equals the number of target faces with the others.
- Many contrasts and subsets are reported; treat single borderline intervals as exploratory.
