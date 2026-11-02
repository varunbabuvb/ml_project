# Analysis summary (auto-generated)

Faces: 1473 | images: 1461 | bootstrap: 2000 resamples of images | seed 42

## Overall balanced accuracy (95% CI)
- raw, 7 classes: Plain 0.428 [0.410, 0.447], Box 0.419 [0.401, 0.438], Box+Number 0.417 [0.398, 0.435], SoV 0.374 [0.355, 0.394]
- raw, without Neutral: Plain 0.500 [0.479, 0.521], Box 0.489 [0.468, 0.511], Box+Number 0.486 [0.465, 0.508], SoV 0.437 [0.414, 0.460]
- self-calibrated, 7 classes: Plain 0.468 [0.445, 0.492], Box 0.483 [0.459, 0.506], Box+Number 0.482 [0.459, 0.505], SoV 0.446 [0.423, 0.469]

## Paired differences, balanced accuracy, points (95% CI)
- **raw, all7**: sov - plain -5.4 [-7.3, -3.5]; box - plain -0.9 [-2.0, +0.2]; box_number - plain -1.2 [-2.3, +0.0]; box_number - box -0.2 [-1.1, +0.7]; sov - box_number -4.2 [-6.0, -2.4]; sov@halfdots - plain -2.8 [-4.4, -1.2]; sov@halfdots - sov +2.6 [+1.1, +4.0]
- **raw, no_neutral**: sov - plain -6.3 [-8.5, -4.1]; box - plain -1.1 [-2.4, +0.2]; box_number - plain -1.3 [-2.7, +0.0]; box_number - box -0.3 [-1.2, +0.8]; sov - box_number -5.0 [-7.0, -2.8]; sov@halfdots - plain -3.4 [-5.2, -1.5]; sov@halfdots - sov +2.9 [+1.2, +4.6]
- **self, all7**: sov - plain -2.2 [-4.4, +0.1]; box - plain +1.4 [-0.0, +2.9]; box_number - plain +1.3 [-0.2, +2.9]; box_number - box -0.1 [-1.2, +0.9]; sov - box_number -3.6 [-5.6, -1.3]; sov@halfdots - plain +0.6 [-1.4, +2.9]; sov@halfdots - sov +2.8 [+1.1, +4.6]
- **self, no_neutral**: sov - plain -0.8 [-3.1, +1.7]; box - plain +1.8 [+0.2, +3.4]; box_number - plain +1.2 [-0.4, +2.9]; box_number - box -0.6 [-1.6, +0.4]; sov - box_number -2.0 [-4.2, +0.2]; sov@halfdots - plain +0.3 [-1.9, +2.6]; sov@halfdots - sov +1.1 [-0.8, +2.9]
- **dev, all7**: sov - plain -1.2 [-3.4, +1.0]
- **dev, no_neutral**: sov - plain -0.2 [-2.6, +2.2]

## SoV - Plain by group (raw, 6 classes; points)
- overall / all faces (n=1473): -7.9 [-9.8, -6.1]
- race_group / Caucasian-like (n=525): -8.2 [-11.2, -5.4]
- race_group / Asian (n=485): -9.3 [-12.7, -6.1]
- race_group / Black (n=463): -6.2 [-9.6, -2.9]
- gender / Male (n=826): -7.3 [-9.8, -4.8]
- gender / Female (n=562): -8.5 [-11.4, -5.6]
- age_band / 0-19 (n=241): -4.6 [-9.2, +0.2]
- age_band / 20-29 (n=520): -10.1 [-13.7, -6.6]
- age_band / 30-39 (n=148): -9.1 [-15.2, -3.3]
- age_band / 40-59 (n=181): -6.0 [-11.2, -0.8]
- age_band / 60+ (n=87, SMALL): -6.7 [-14.4, -0.5]
- face_size / small (n=531): -11.3 [-15.0, -8.0]
- face_size / medium (n=458): -6.1 [-9.4, -2.7]
- face_size / large (n=484): -6.8 [-9.8, -3.8]
- crop_type / single-face (n=1205): -8.4 [-10.5, -6.3]
- crop_type / multi-face (n=268): -3.8 [-7.6, +0.0]

## Is the SoV - Plain effect different between groups? (raw, 6 classes; difference vs reference level)
- race_group: Asian vs Caucasian-like: -1.1 [-5.3, +2.8] -> no clear difference
- race_group: Black vs Caucasian-like: +2.0 [-2.5, +6.3] -> no clear difference
- gender: Female vs Male: -1.2 [-4.9, +2.7] -> no clear difference
- age_band: 0-19 vs 20-29: +5.5 [+0.0, +11.3] -> CI excludes 0
- age_band: 30-39 vs 20-29: +1.0 [-5.7, +7.7] -> no clear difference
- age_band: 40-59 vs 20-29: +4.2 [-2.2, +10.4] -> no clear difference
- age_band: 60+ vs 20-29: +3.4 [-4.7, +10.6] -> no clear difference
- face_size: medium vs small: +5.2 [+0.5, +10.3] -> CI excludes 0
- face_size: large vs small: +4.6 [+0.2, +9.1] -> CI excludes 0
- crop_type: multi-face vs single-face: +4.6 [+0.2, +8.7] -> CI excludes 0

## Numbered vs centre prompt on multi-face crops (accuracy)
- box_number (n=268): numbered 0.437 vs centre 0.444, diff -0.7 [-3.0, +1.3]
- sov (n=268): numbered 0.418 vs centre 0.425, diff -0.7 [-3.2, +1.8]
- sov@halfdots (n=268): numbered 0.429 vs centre 0.440, diff -1.1 [-3.2, +0.7]

## Warnings
- Fear is excluded from per-group results (too few faces); the model predicts Neutral almost never, so Neutral recall is ~0 in every condition (see the no_neutral rows).
- Group labels are FairFace ESTIMATES; gender/age use only faces whose label was stable across two crops.
- Group x emotion cells below 30 faces: Black/fear=13
