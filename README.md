# realistec-stereo

A high-resolution-capable fork of [Google's FILM frame interpolation model](https://github.com/google-research/frame-interpolation), packaged as a [Cog](https://github.com/replicate/cog) model for [Replicate](https://replicate.com).

## What this is

[FILM (Frame Interpolation for Large Motion)](https://film-net.github.io/) is a state-of-the-art neural network for generating in-between frames from a pair of input images. The [upstream Replicate model](https://replicate.com/google-research/frame-interpolation) (`google-research/frame-interpolation`) is hard-capped at roughly 1920x1080 input resolution because its `predict.py` wrapper does not expose FILM's built-in patch-subdivision parameters.

This fork adds:

- **`block_height`** / **`block_width`** — patch subdivision, so inputs above 1920x1080 can be interpolated without downscaling
- **`num_views`** — return an evenly spaced subset of the generated frames as lossless PNGs
- **`preview_short_edge`** / **`share_short_edge`** — sizes of the two preview artifacts

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
| `preview_short_edge` | integer | 1080 | 240-2160 | Short-edge size of the preview MP4. Never upscales. Does not affect the PNGs. |
| `share_short_edge` | integer | 480 | 160-1080 | Short-edge size of the shareable animated WebP. Never upscales. |

## Output

An object with three fields, all file URLs:

| Field | Contents |
| --- | --- |
| `preview` | H.264 MP4 at 30 fps, a forward pass through **every** generated frame, sized to `preview_short_edge`. A review artifact — opened in a player with a scrubber, where a straight pass is easiest to step through. |
| `share` | Animated WebP, **ping-pong** loop of the same frames, sized to `share_short_edge`, looping infinitely. |
| `frames` | A zip of exactly `num_views` lossless PNGs at full working resolution, plus a `manifest.json`. |

```json
{
  "preview": "https://replicate.delivery/.../preview.mp4",
  "share":   "https://replicate.delivery/.../share.webp",
  "frames":  "https://replicate.delivery/.../frames.zip"
}
```

### Why the share artifact is a WebP, and why it ping-pongs

WebP is an **image** format, so it renders in an `<img>` tag and loops silently with no player chrome. An MP4 opened standalone is handed to a video player, which shows transport controls that reappear on every loop — fine in a page you control, unavoidable when someone opens a downloaded file.

Only the WebP ping-pongs (`0..n-1` then `n-2..1`). It autoplays with no scrubber and no way to reverse by hand, so the loop itself has to carry the back-and-forth; a forward loop would jump-cut from the last frame straight back to the first, which is the largest single jump in the sequence. Endpoints are not repeated — doing so holds the first and last frame for two frame periods and reads as a stutter at each turn. The MP4 has a scrubber, so it just plays forward.

**It is not a GIF because GIF is far larger.** Measured on a 128-frame ping-pong at 480 short edge:

| Format | Size |
| --- | --- |
| MP4 H.264 | 0.24 MB |
| **Animated WebP** | **1.75 MB** |
| GIF, 2-pass palette | 10.29 MB |
| GIF, naive palette | 23.18 MB |

GIF also caps at 256 colours, which bands visibly on photographic content. WebP gives full colour at roughly a sixth of the size and loops identically.

### A note on short-edge sizing

Sizing by the short edge is **not** a size saving over long-edge sizing — for a 3:2 frame, short edge 1080 gives 1620x1080 where long edge 1280 gave 1280x853, about 60% more pixels. It is chosen for the guarantee: the narrow dimension holds up whatever the orientation, where a long-edge cap leaves portrait frames narrow and landscape frames short.

There is deliberately no long-edge cap. Scaling by the short edge is unbounded on the other axis in principle, but inputs to this model are expected to be pre-cropped to a bounded aspect ratio. Add a cap if you intend to feed it panoramic material.

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

Preview frames are forced to even pixel dimensions for the MP4 on every path, including when no downscaling occurs. libx264 with `yuv420p` rejects odd dimensions, and that would otherwise surface as an ffmpeg failure at the very end of a long prediction. The WebP has no such constraint, so it is not rounded there.

The ping-pong tail costs no extra memory: it is the same frames indexed in reverse. The share frames are derived from the stored preview frames at write time rather than accumulated in a second list.

## Changes in v2.2

**Breaking.** `preview_long_edge` was replaced by `preview_short_edge` (default 1080) and `share_short_edge` (default 480). Callers passing `preview_long_edge` will get an error.

- New `share` output: an animated WebP that ping-pongs and loops forever.
- The MP4 is now sized by short edge, so it is larger than before at default settings.

## Changes in v2.1

- Fixed a hang in `image_to_patches` / `patches_to_image`. Both split along an axis whose length is one patch's **pixel count** rather than the patch count — 2,300,881 tensors for a 1919x2399 input at a 2x1 grid — so any input large enough to need block subdivision ran indefinitely without erroring. Only `1x1` inputs ever completed, because `_interpolate_with_blocks()` returns early before reaching that code. Replaced with reshape + transpose, verified byte-identical to the original.

This means the block subdivision grids this fork exists to provide did not work before v2.1.

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
