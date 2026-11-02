# Multi-face analysis summary (auto-generated)

Target faces with results: 2816 of 2818 | images: 1250 (mixed: 428) | chunks present: [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5)]... (32 of 32) | bootstrap 2000 resamples of images

## Accuracy per arm (95% CI)
- **all / acc_all**: Plain (position) 0.657 [0.634, 0.679] (n=2816); Box (position) 0.658 [0.636, 0.681] (n=2816); Box+Number (number) 0.663 [0.640, 0.685] (n=2816); SoV (number) 0.645 [0.622, 0.667] (n=2816); Box+Number (position) 0.425 [0.400, 0.448] (n=997); SoV (position) 0.389 [0.364, 0.414] (n=997)
- **mixed / acc_all**: Plain (position) 0.422 [0.398, 0.446] (n=997); Box (position) 0.425 [0.400, 0.450] (n=997); Box+Number (number) 0.434 [0.409, 0.457] (n=997); SoV (number) 0.410 [0.385, 0.435] (n=997); Box+Number (position) 0.425 [0.400, 0.448] (n=997); SoV (position) 0.389 [0.364, 0.414] (n=997)
- **all / acc_expr**: Plain (position) 0.336 [0.269, 0.402] (n=301); Box (position) 0.312 [0.244, 0.376] (n=301); Box+Number (number) 0.289 [0.231, 0.347] (n=301); SoV (number) 0.166 [0.118, 0.214] (n=301); Box+Number (position) 0.236 [0.179, 0.294] (n=225); SoV (position) 0.133 [0.089, 0.184] (n=225)
- **mixed / xref**: Plain (position) 0.339 [0.316, 0.364] (n=997); Box (position) 0.345 [0.323, 0.369] (n=997); Box+Number (number) 0.329 [0.307, 0.352] (n=997); SoV (number) 0.358 [0.335, 0.381] (n=997); Box+Number (position) 0.329 [0.306, 0.351] (n=997); SoV (position) 0.327 [0.302, 0.352] (n=997)

## Paired differences, points (95% CI)
- **all / acc_all**
    - SoV as designed vs Plain (sov|numbered - plain|positional): -1.2 [-2.2, -0.3]  *
    - Box+Number as designed vs Plain (box_number|numbered - plain|positional): +0.6 [-0.0, +1.2]
    - Boxes alone vs Plain (same prompt) (box|positional - plain|positional): +0.1 [-0.5, +0.7]
    - Numbers + numbered prompt vs boxes (box_number|numbered - box|positional): +0.5 [-0.1, +1.1]
    - Landmarks (same overlay type, same prompt) (sov|numbered - box_number|numbered): -1.8 [-2.6, -1.1]  *
    - Using the number vs counting (same image) (box_number|numbered - box_number|positional): +0.9 [-0.1, +1.9]
    - Using the number vs counting (SoV image) (sov|numbered - sov|positional): +2.1 [+1.0, +3.3]  *
    - Number tags themselves (same prompt) (box_number|positional - box|positional): +0.0 [-1.2, +1.2]
    - SoV marks only (same prompt) vs Plain (sov|positional - plain|positional): -3.3 [-5.2, -1.5]  *
- **mixed / acc_all**
    - SoV as designed vs Plain (sov|numbered - plain|positional): -1.2 [-3.2, +0.6]
    - Box+Number as designed vs Plain (box_number|numbered - plain|positional): +1.2 [-0.3, +2.7]
    - Boxes alone vs Plain (same prompt) (box|positional - plain|positional): +0.3 [-0.8, +1.3]
    - Numbers + numbered prompt vs boxes (box_number|numbered - box|positional): +0.9 [-0.6, +2.4]
    - Landmarks (same overlay type, same prompt) (sov|numbered - box_number|numbered): -2.4 [-3.8, -1.0]  *
    - Using the number vs counting (same image) (box_number|numbered - box_number|positional): +0.9 [-0.1, +1.9]
    - Using the number vs counting (SoV image) (sov|numbered - sov|positional): +2.1 [+1.0, +3.3]  *
    - Number tags themselves (same prompt) (box_number|positional - box|positional): +0.0 [-1.2, +1.2]
    - SoV marks only (same prompt) vs Plain (sov|positional - plain|positional): -3.3 [-5.2, -1.5]  *
- **mixed / xref**
    - SoV as designed vs Plain (sov|numbered - plain|positional): +1.9 [-0.5, +4.1]
    - Box+Number as designed vs Plain (box_number|numbered - plain|positional): -1.0 [-3.1, +0.9]
    - Boxes alone vs Plain (same prompt) (box|positional - plain|positional): +0.6 [-0.7, +1.9]
    - Numbers + numbered prompt vs boxes (box_number|numbered - box|positional): -1.6 [-3.4, +0.0]
    - Landmarks (same overlay type, same prompt) (sov|numbered - box_number|numbered): +2.9 [+1.1, +4.6]  *
    - Using the number vs counting (same image) (box_number|numbered - box_number|positional): +0.0 [-1.3, +1.4]
    - Using the number vs counting (SoV image) (sov|numbered - sov|positional): +3.1 [+1.8, +4.4]  *
    - Number tags themselves (same prompt) (box_number|positional - box|positional): -1.6 [-3.0, -0.4]  *
    - SoV marks only (same prompt) vs Plain (sov|positional - plain|positional): -1.2 [-3.6, +1.0]
- **all / acc_expr**
    - SoV as designed vs Plain (sov|numbered - plain|positional): -16.9 [-23.1, -10.9]  *
    - Box+Number as designed vs Plain (box_number|numbered - plain|positional): -4.7 [-9.2, +0.0]
    - Boxes alone vs Plain (same prompt) (box|positional - plain|positional): -2.3 [-6.2, +1.0]
    - Numbers + numbered prompt vs boxes (box_number|numbered - box|positional): -2.3 [-6.0, +1.1]
    - Landmarks (same overlay type, same prompt) (sov|numbered - box_number|numbered): -12.3 [-16.9, -7.9]  *
    - Using the number vs counting (same image) (box_number|numbered - box_number|positional): -2.7 [-5.6, +0.0]
    - Using the number vs counting (SoV image) (sov|numbered - sov|positional): -1.8 [-4.3, +0.5]
    - Number tags themselves (same prompt) (box_number|positional - box|positional): +0.0 [-3.8, +3.6]
    - SoV marks only (same prompt) vs Plain (sov|positional - plain|positional): -11.6 [-17.9, -6.1]  *
- **not_mixed / acc_all**
    - SoV as designed vs Plain (sov|numbered - plain|positional): -1.2 [-2.3, -0.3]  *
    - Box+Number as designed vs Plain (box_number|numbered - plain|positional): +0.3 [-0.3, +0.8]
    - Boxes alone vs Plain (same prompt) (box|positional - plain|positional): +0.1 [-0.6, +0.6]
    - Numbers + numbered prompt vs boxes (box_number|numbered - box|positional): +0.2 [-0.2, +0.6]
    - Landmarks (same overlay type, same prompt) (sov|numbered - box_number|numbered): -1.5 [-2.3, -0.7]  *

## Does the SoV-vs-Plain effect differ between groups? (acc_all, 'SoV as designed vs Plain'; * = CI excludes 0)
- race_group: Asian vs Caucasian-like: +1.1 [-1.0, +3.0]
- race_group: Black vs Caucasian-like: +1.2 [-1.3, +3.8]
- gender: Male vs Female: -0.0 [-1.5, +1.5]
- age_band: 0-19 vs 20-29: -0.6 [-3.0, +1.8]
- age_band: 30-39 vs 20-29: +0.4 [-2.8, +3.3]
- age_band: 40-59 vs 20-29: +0.3 [-2.4, +3.1]
- age_band: 60+ vs 20-29: -0.9 [-6.1, +4.3]
- face_size: medium vs small: +0.7 [-1.9, +3.3]
- face_size: large vs small: +2.0 [-0.3, +4.6]
- image_type: mixed vs not_mixed: +0.0 [-2.2, +2.0]
- reference_clear: extra/missing detections vs detections = targets: -0.6 [-2.8, +1.3]
- n_targets: 3+ faces vs 2 faces: -1.5 [-4.3, +1.0]

## Warnings
- Emotions are very unbalanced (mostly Happy and Neutral; Neutral is never predicted). acc_all mostly measures Happy; use acc_expr and xref for the informative faces.
- Group labels are FairFace ESTIMATES. Beyond Happy, the group cells are too small for group claims; cells below 30 faces: Asian/angry=5, Asian/disgust=13, Asian/fear=0, Asian/sad=20, Asian/surprise=17, Black/angry=0, Black/disgust=11, Black/fear=0, Black/sad=4, Black/surprise=7, Caucasian-like/fear=9
- Positional prompts make plain/box rely on the model counting faces; 'reference_clear' rows compare images where the number of detected faces equals the number of target faces with the others.
- Many contrasts and subsets are reported; treat single borderline intervals as exploratory.
