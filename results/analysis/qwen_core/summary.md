# Analysis summary (auto-generated)

Faces: 1473 | images: 1461 | bootstrap: 2000 resamples of images | seed 42

## Overall balanced accuracy (95% CI)
- raw, 7 classes: Plain 0.493 [0.473, 0.515], Box 0.492 [0.472, 0.512], Box+Number 0.498 [0.477, 0.519], SoV 0.412 [0.393, 0.432]
- raw, without Neutral: Plain 0.423 [0.400, 0.447], Box 0.426 [0.403, 0.450], Box+Number 0.433 [0.410, 0.457], SoV 0.322 [0.301, 0.345]
- self-calibrated, 7 classes: Plain 0.514 [0.492, 0.538], Box 0.514 [0.492, 0.537], Box+Number 0.509 [0.487, 0.532], SoV 0.448 [0.428, 0.471]

## Paired differences, balanced accuracy, points (95% CI)
- **raw, all7**: sov - plain -8.1 [-10.0, -6.2]; box - plain -0.2 [-1.6, +1.2]; box_number - plain +0.5 [-1.0, +1.8]; box_number - box +0.6 [-0.3, +1.5]; sov - box_number -8.5 [-10.5, -6.6]; sov@halfdots - plain -4.7 [-6.4, -3.0]; sov@halfdots - sov +3.4 [+1.8, +4.8]
- **raw, no_neutral**: sov - plain -10.0 [-12.1, -7.9]; box - plain +0.3 [-1.2, +1.8]; box_number - plain +1.0 [-0.4, +2.5]; box_number - box +0.7 [-0.2, +1.7]; sov - box_number -11.1 [-13.2, -9.0]; sov@halfdots - plain -5.7 [-7.6, -3.8]; sov@halfdots - sov +4.4 [+2.7, +6.0]
- **self, all7**: sov - plain -6.5 [-8.8, -4.3]; box - plain +0.1 [-1.6, +1.7]; box_number - plain -0.5 [-2.2, +1.1]; box_number - box -0.6 [-1.6, +0.4]; sov - box_number -6.0 [-8.3, -3.8]; sov@halfdots - plain -4.6 [-6.7, -2.5]; sov@halfdots - sov +2.0 [+0.3, +3.7]
- **self, no_neutral**: sov - plain -8.1 [-10.4, -5.7]; box - plain +0.7 [-1.1, +2.4]; box_number - plain +0.1 [-1.7, +1.8]; box_number - box -0.6 [-1.7, +0.4]; sov - box_number -8.1 [-10.5, -5.7]; sov@halfdots - plain -5.5 [-7.8, -3.3]; sov@halfdots - sov +2.6 [+0.7, +4.4]

## SoV - Plain by group (raw, 6 classes; points)
- overall / all faces (n=1473): -9.0 [-11.2, -6.9]
- race_group / Caucasian-like (n=525): -9.1 [-13.2, -5.1]
- race_group / Asian (n=485): -13.6 [-17.3, -9.8]
- race_group / Black (n=463): -4.4 [-7.9, -0.9]
- gender / Male (n=826): -10.3 [-13.2, -7.3]
- gender / Female (n=562): -7.1 [-10.7, -3.5]
- age_band / 0-19 (n=241): -8.1 [-14.5, -1.3]
- age_band / 20-29 (n=520): -10.5 [-14.1, -6.9]
- age_band / 30-39 (n=148): -7.5 [-14.7, -0.5]
- age_band / 40-59 (n=181): -3.5 [-9.4, +2.1]
- age_band / 60+ (n=87, SMALL): -13.1 [-22.4, -4.3]
- face_size / small (n=531): -8.9 [-12.7, -5.3]
- face_size / medium (n=458): -9.3 [-13.8, -5.2]
- face_size / large (n=484): -7.9 [-11.8, -3.9]
- crop_type / single-face (n=1205): -8.6 [-11.0, -6.2]
- crop_type / multi-face (n=268): -9.4 [-15.7, -3.5]

## Is the SoV - Plain effect different between groups? (raw, 6 classes; difference vs reference level)
- race_group: Asian vs Caucasian-like: -4.4 [-9.9, +1.0] -> no clear difference
- race_group: Black vs Caucasian-like: +4.7 [-0.5, +10.0] -> no clear difference
- gender: Female vs Male: +3.1 [-1.6, +8.1] -> no clear difference
- age_band: 0-19 vs 20-29: +2.4 [-4.7, +10.1] -> no clear difference
- age_band: 30-39 vs 20-29: +3.0 [-5.4, +11.3] -> no clear difference
- age_band: 40-59 vs 20-29: +7.0 [+0.2, +14.0] -> CI excludes 0
- age_band: 60+ vs 20-29: -2.5 [-12.8, +7.0] -> no clear difference
- face_size: medium vs small: -0.4 [-6.3, +5.4] -> no clear difference
- face_size: large vs small: +1.0 [-4.1, +6.5] -> no clear difference
- crop_type: multi-face vs single-face: -0.8 [-7.8, +5.6] -> no clear difference

## Numbered vs centre prompt on multi-face crops (accuracy)
- box_number (n=268): numbered 0.530 vs centre 0.504, diff +2.6 [+0.0, +5.3]
- sov (n=268): numbered 0.437 vs centre 0.429, diff +0.7 [-2.4, +4.0]
- sov@halfdots (n=268): numbered 0.485 vs centre 0.463, diff +2.2 [+0.0, +4.9]

## Warnings
- Fear is excluded from per-group results (too few faces); the model predicts Neutral almost never, so Neutral recall is ~0 in every condition (see the no_neutral rows).
- Group labels are FairFace ESTIMATES; gender/age use only faces whose label was stable across two crops.
- Group x emotion cells below 30 faces: Black/fear=13
