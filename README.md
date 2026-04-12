# realistec-stereo

A high-resolution-capable fork of [Google's FILM frame interpolation model](https://github.com/google-research/frame-interpolation), packaged as a [Cog](https://github.com/replicate/cog) model for [Replicate](https://replicate.com).

## What this is

[FILM (Frame Interpolation for Large Motion)](https://film-net.github.io/) is a state-of-the-art neural network for generating in-between frames from a pair of input images. The [upstream Replicate model](https://replicate.com/google-research/frame-interpolation) (`google-research/frame-interpolation`) is hard-capped at roughly 1920x1080 input resolution because its `predict.py` wrapper does not expose FILM's built-in patch-subdivision parameters.

This fork adds two parameters that the upstream model does not:

- **`block_height`** — number of rows to subdivide the input frames into (1-8, default 1)
- **`block_width`** — number of columns to subdivide the input frames into (1-8, default 1)

When both are 1 (the default), behavior is identical to the upstream model. When either is greater than 1, the input frames are folded into `block_height * block_width` non-overlapping patches, the model is run on each patch pair, and the results are reassembled into a full-resolution output frame. This makes it possible to interpolate at 4K (3840x2160) and beyond without downscaling.

## Inputs

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `frame1` | image | required | — | First input frame (PNG, JPG, or JPEG) |
| `frame2` | image | required | — | Second input frame (PNG, JPG, or JPEG) |
| `times_to_interpolate` | integer | 1 | 1-8 | If 1, output is a single PNG mid-frame at t=0.5. If > 1, output is an MP4 at 30 fps with 2^N + 1 frames. |
| `block_height` | integer | 1 | 1-8 | Patch subdivision rows |
| `block_width` | integer | 1 | 1-8 | Patch subdivision columns |

## Output

- A PNG file when `times_to_interpolate == 1`
- An MP4 video when `times_to_interpolate > 1`

## Block subdivision sizing

For a `block_height x block_width` grid, each patch is roughly `(input_height / block_height) x (input_width / block_width)` pixels. FILM was designed to work well on inputs around 1920x1080, so size your block grid to keep each patch in that ballpark:

| Input resolution | Recommended grid | Patch size |
|---|---|---|
| 1920x1080 (1080p) | 1x1 | 1920x1080 |
| 2560x1440 (1440p) | 2x2 | 1280x720 |
| 3840x2160 (4K) | 2x2 | 1920x1080 |
| 7680x4320 (8K) | 4x4 | 1920x1080 |

Each patch adds prediction time roughly linearly, so a 2x2 grid takes about 4x as long as a 1x1 prediction.

## How it works

The model is FILM's stock pretrained `Style` SavedModel (the same one Google publishes on TF Hub as [`film/1`](https://www.kaggle.com/models/google/film)). The patch-folding helpers (`_pad_to_align`, `image_to_patches`, `patches_to_image`) are copied verbatim from Google Research's [`eval/interpolator.py`](https://github.com/google-research/frame-interpolation/blob/main/eval/interpolator.py). The Cog wrapper logic around them is original, written for this model.

## License

The Cog wrapper code in `predict.py` is released under the Apache License 2.0, the same license as the upstream FILM model. The bundled FILM SavedModel is copyright 2022 Google LLC, also Apache 2.0 licensed.

## Citation

If you use this model in research or production, please cite the original FILM paper:

```
@inproceedings{reda2022film,
 title = {FILM: Frame Interpolation for Large Motion},
 author = {Fitsum Reda and Janne Kontkanen and Eric Tabellion and Deqing Sun and Caroline Pantofaru and Brian Curless},
 booktitle = {European Conference on Computer Vision (ECCV)},
 year = {2022}
}
```
