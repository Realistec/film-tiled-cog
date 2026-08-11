# realistec-stereo

A high-resolution-capable fork of [Google's FILM frame interpolation model](https://github.com/google-research/frame-interpolation), packaged as a [Cog](https://github.com/replicate/cog) model for [Replicate](https://replicate.com).

## What this is

[FILM (Frame Interpolation for Large Motion)](https://film-net.github.io/) is a state-of-the-art neural network for generating in-between frames from a pair of input images. The [upstream Replicate model](https://replicate.com/google-research/frame-interpolation) (`google-research/frame-interpolation`) is hard-capped at roughly 1920x1080 input resolution because its `predict.py` wrapper does not expose FILM's built-in patch-subdivision parameters.

This fork adds:

- **`block_height`** / **`block_width`** — patch subdivision, so inputs above 1920x1080 can be interpolated without downscaling
- **`num_views`** — return an evenly spaced subset of the generated frames as lossless PNGs
- **`preview_long_edge`** — size of the preview video

It targets workflows where interpolated frames are **inputs to further image processing** rather than something to be watched, so pixel-accurate output matters more than a compact video file.

## Inputs

| Parameter | Type | Default | Range | Description |
| --- | --- | --- | --- | --- |
| `frame1` | image | required | — | First input frame (PNG, JPG, or JPEG) |
| `frame2` | image | required | — | Second input frame (PNG, JPG, or JPEG) |
| `times_to_interpolate` | integer | 1 | 1-8 | Recursion depth. Produces `2^N + 1` frames. See [Frame selection](#frame-selection) before changing this. |
| `block_height` | integer | 1 | 1-8 | Patch subdivision rows |
| `block_width` | integer | 1 | 1-8 | Patch subdivision columns |
| `num_views` | integer | 12 | 2-257 | How many evenly spaced frames to return as lossless PNGs, first and last pinned to the input frames. Clamped to the number of frames actually generated. |
| `preview_long_edge` | integer | 1280 | 240-3840 | Long-edge cap for the preview MP4. Never upscales. Does not affect the PNGs. |

## Output

An object with two fields, both file URLs:

| Field | Contents |
| --- | --- |
| `preview` | H.264 MP4 at 30 fps containing **every** generated frame, downscaled to `preview_long_edge`. A quick visual check, not a deliverable. |
| `frames` | A zip of exactly `num_views` lossless PNGs at full working resolution, plus a `manifest.json`. |

```json
{
  "preview": "https://replicate.delivery/.../preview.mp4",
  "frames":  "https://replicate.delivery/.../frames.zip"
}
```

Output files are deleted roughly an hour after the prediction for API-created predictions, so consumers must copy them rather than link to them.

### Zip contents

```
view_00_src000.png
view_01_src006.png
view_02_src012.png
...
view_11_src064.png
manifest.json
```

Each filename carries both the **output ordinal** and the **true source frame index**. Selected indices are non-contiguous and unevenly spaced by necessity, so a bare sequence number would lose the information needed to verify a selection.

Entries are stored uncompressed (`ZIP_STORED`). PNG is already deflate-compressed; re-deflating measured a 6 KB gain across 172 MB of frames, which does not justify the CPU time.

`manifest.json`:

```json
{
  "schema": 1,
  "num_views": 12,
  "total_frames": 65,
  "times_to_interpolate": 6,
  "block_height": 1,
  "block_width": 1,
  "frame_width": 4000,
  "frame_height": 4000,
  "views": [
    { "view": 0, "source_index": 0,  "file": "view_00_src000.png" },
    { "view": 1, "source_index": 6,  "file": "view_01_src006.png" }
  ]
}
```

### Why PNG and not a video

H.264 is lossy in ways that matter when frames feed further processing rather than being watched. 4:2:0 chroma subsampling stores colour at half resolution on both axes, and DCT quantisation concentrates its error on fine high-frequency detail. Measured on a synthetic 65-frame sequence at CRF 18, mean absolute error was **7.39 on fine vertical lines against 2.45 elsewhere** — a 3x concentration on the detail most likely to matter downstream.

Formats measured on the same sequence (65 frames, 1920x1280):

| Format | Size | Round-trips bit-exact? |
| --- | --- | --- |
| PNG sequence | 172 MB | reference |
| FFV1 (RGB) | 152 MB | yes — max error 0 |
| x264rgb `-qp 0` | 146 MB | yes — max error 0 |
| ProRes 4444 | 106 MB | **no** — max error 21/255 |
| H.264 4:2:0 CRF 18 | 1.1 MB | no — max error 197/255 |

Two results worth noting. ProRes 4444 is *not* mathematically lossless despite the name — "4444" describes chroma sampling, not fidelity. And inter-frame prediction buys almost nothing here: x264rgb's 146 MB against FFV1's 152 MB is a 4% gain, because per-frame sensor noise is uncorrelated between frames and defeats temporal prediction.

## Frame selection

`num_views` frames are selected at evenly spaced indices with the first and last **pinned to the two input frames**, since those are the only real source images — dropping either would discard captured detail in favour of an interpolated approximation of it.

Selection is derived from the frame count actually produced, not from a hardcoded 65, so changing `times_to_interpolate` moves the selection with it rather than sampling the wrong positions.

### Do not lower `times_to_interpolate` to "save work"

Interpolating 65 frames and keeping 12 looks wasteful. It is not.

Selected frames must be evenly spaced in time, or the step between consecutive outputs varies. Neither 11 nor 9 intervals divides any power of two, so **exact even spacing is impossible at any depth** — the only question is how uneven. Measured spread of gap sizes when selecting 12 frames:

| `times_to_interpolate` | Frames | GPU cost | Gaps | Spread |
| --- | --- | --- | --- | --- |
| 3 | 9 | 0.12x | `1 0 1 1 1 0 1 …` | 138% |
| 4 | 17 | 0.25x | `1 2 1 2 1 2 …` | 69% |
| 5 | 33 | 0.50x | `3 3 3 3 3 2 …` | 34% |
| **6** | **65** | **1.00x** | `6 6 5 6 6 6 …` | **17%** |
| 7 | 129 | 2.00x | `12 11 12 12 …` | 9% |

At depth 3 some gaps reach **zero** — two byte-identical frames in the output, which a consumer has no way to detect. Depth 6 is the last one at 1x GPU cost and is the intended operating point.

Counts that divide a power of two evenly are exempt from all of this: 5 frames from 65 gives gaps of exactly 16, and 9, 17 or 33 frames divide cleanly too.

## Memory behaviour

Frames are written to disk as they are yielded and released immediately. **Do not reintroduce `list(self._recursive_interpolate(...))`.**

The v1 predictor built the full frame list in memory and then built a uint8 copy alongside it, holding both at once:

| Input size | Peak host RAM (v1) | Peak host RAM (v2) |
| --- | --- | --- |
| 1920x1280 | 2.2 GB | ~0.5 GB |
| 3600x3600 | 11.8 GB | ~1.2 GB |
| 4000x4000 | 14.5 GB | ~1.4 GB |
| 5464x5464 | 27.1 GB | ~2.5 GB |

The v1 path could not complete a full-resolution job at all.

## Block subdivision sizing

For a `block_height x block_width` grid, each patch is roughly `(input_height / block_height) x (input_width / block_width)` pixels. FILM was designed to work well on inputs around 1920x1080, so size the grid to keep each patch in that ballpark:

| Input resolution | Recommended grid | Patch size |
| --- | --- | --- |
| 1920x1080 (1080p) | 1x1 | 1920x1080 |
| 2560x1440 (1440p) | 2x2 | 1280x720 |
| 3840x2160 (4K) | 2x2 | 1920x1080 |
| 7680x4320 (8K) | 4x4 | 1920x1080 |

Each patch adds prediction time roughly linearly, so a 2x2 grid takes about 4x as long as a 1x1 prediction.

## How it works

The model is FILM's stock pretrained `Style` SavedModel (the same one Google publishes on TF Hub as [`film/1`](https://www.kaggle.com/models/google/film)), bundled into the image at build time rather than fetched at runtime, which keeps builds reproducible and avoids a model pull on every scale-from-zero.

The patch-folding helpers (`_pad_to_align`, `image_to_patches`, `patches_to_image`) are copied verbatim from Google Research's [`eval/interpolator.py`](https://github.com/google-research/frame-interpolation/blob/main/eval/interpolator.py). The Cog wrapper logic around them is original, written for this model.

Preview frames are forced to even pixel dimensions on every path, including when no downscaling occurs. libx264 with `yuv420p` rejects odd dimensions, and that would otherwise surface as an ffmpeg failure at the very end of a long prediction.

## Changes in v2

**Breaking.** The return type changed from a single file to an object.

- Output is now `{ preview, frames }` instead of a bare PNG or MP4.
- The `times_to_interpolate == 1` single-PNG branch was **removed**. It returned a bare file path, which cannot coexist with an object return type. That case now falls through to the normal path and yields a 3-frame sequence. This is the one place the fork no longer matches upstream behaviour.
- New inputs `num_views` and `preview_long_edge`.
- Frames stream to disk instead of accumulating in memory.

Consumers pinned to an older version hash are unaffected until they update the pin.

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
