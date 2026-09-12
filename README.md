# realistec-multi

A multi-input, high-resolution-capable fork of [Google's FILM frame interpolation model](https://github.com/google-research/frame-interpolation), packaged as a [Cog](https://github.com/replicate/cog) model for [Replicate](https://replicate.com).

> **Branch note.** The deployed model is built from **`phase1-multi-frame`**, which has never been merged to `master`. `master` may still carry the two-frame `realistec-stereo` `predict.py` and its push target. Check which branch you are on before running `cog push`, or you will overwrite the multi-frame model with the stereo one.

## What this is

[FILM (Frame Interpolation for Large Motion)](https://film-net.github.io/) generates in-between frames from a pair of input images. The [upstream Replicate model](https://replicate.com/google-research/frame-interpolation) takes exactly two frames and is hard-capped at roughly 1920x1080, because its wrapper does not expose FILM's patch-subdivision parameters.

This fork changes two things:

- **Three or four input frames**, interpolated across the whole run as one continuous sequence rather than as separate pairs.
- **`block_height` / `block_width`**, which subdivide each frame into non-overlapping patches so inputs far above 1920x1080 can be processed without downscaling.

It targets workflows where the interpolated frames are **inputs to further image processing** — here, lenticular print production — rather than something to be watched. Pixel accuracy matters more than a compact video file.

With `block_height` and `block_width` both at 1, patch behaviour is identical to upstream.

## Segments

Consecutive input frames form **segments**. Three inputs give two segments, four give three. Each segment is interpolated independently to depth `times_to_interpolate`, and the results are concatenated:

```
total_frames = segments * 2**times_to_interpolate + 1
```

At the fixed depth of 5 that is **65 frames from three inputs** and **97 from four**.

## Inputs

| Parameter | Type | Default | Range | Description |
| --- | --- | --- | --- | --- |
| `frame1` | image | required | — | First input frame (PNG, JPG, or JPEG) |
| `frame2` | image | required | — | Second input frame |
| `frame3` | image | required | — | Third input frame |
| `frame4` | image | *optional* | — | Fourth input frame. Adds a third segment, raising the sequence from 65 to 97 frames and tightening view spacing. **Omit the key entirely** for a three-frame job — do not send `null`. |
| `times_to_interpolate` | integer | 5 | 1-8 | Recursion depth **per segment**. Leave at 5 — see [Frame selection](#frame-selection). |
| `block_height` | integer | 1 | 1-8 | Patch subdivision rows |
| `block_width` | integer | 1 | 1-8 | Patch subdivision columns |
| `num_views` | integer | 12 | 2-257 | How many evenly spaced frames to return as lossless PNGs. Clamped to the number actually generated. |
| `preview_short_edge` | integer | 1080 | 240-2160 | Short-edge size of the preview MP4. Never upscales. Does not affect the PNGs. |
| `share_short_edge` | integer | 480 | 160-1080 | Short-edge size of the animated WebP. Never upscales. |

### `frame4` is `Optional[Path]`, and that is load-bearing

`frame4` is declared `Optional[Path]` **with** `default=None`. Both parts, together.

An earlier version of this file used a bare `Path` and carried a comment asserting that `Optional[X]` would be rejected at build time by cog's `validate_input_type()`. **That is wrong**, and the identical comment on the sibling depth model `da3mono-large-multi` caused a production failure: Replicate advertised `frame4` as **required** in the OpenAPI schema and answered every three-frame prediction with

```
422 — input: frame4 is required
```

before the model code ever ran. Four-frame predictions were unaffected, so it looked healthy for a week.

**This model has not failed that way yet, and the reason matters.** The deployed image predates the cog runtime that enforces it. `cog.yaml` pins no cog version, so the next `cog push` builds against whatever is current — and a bare `Path` would start returning 422 on every three-frame order, after the customer has paid.

Before changing this declaration, check the published schema:

```
GET /v1/models/realistecsales/realistec-multi/versions/<hash>
  -> openapi_schema.components.schemas.Input.required
```

`frame4` must **not** appear in that array.

## Output

An object with three file URLs:

| Field | Contents |
| --- | --- |
| `preview` | H.264 MP4, forward pass through **every** generated frame, sized to `preview_short_edge`. A review artifact — opened in a player with a scrubber, where a straight pass is easiest to step through. |
| `share` | Animated WebP, **ping-pong** loop of the same frames, sized to `share_short_edge`, looping forever. |
| `frames` | A zip of exactly `num_views` lossless PNGs at full working resolution, plus a `manifest.json`. |

```json
{
  "preview": "https://replicate.delivery/.../preview.mp4",
  "share":   "https://replicate.delivery/.../share.webp",
  "frames":  "https://replicate.delivery/.../frames.zip"
}
```

Output files are deleted roughly an hour after the prediction for API-created predictions, so consumers must copy them rather than link to them.

### Why the share artifact is a WebP, and why it ping-pongs

WebP is an **image** format, so it renders in an `<img>` tag and loops silently with no player chrome. An MP4 opened standalone is handed to a video player, which shows transport controls that reappear on every loop.

Only the WebP ping-pongs (`0..n-1` then `n-2..1`). It autoplays with no scrubber and no way to reverse by hand, so the loop itself has to carry the back-and-forth — a forward loop would jump-cut from the last view straight back to the first, the widest parallax jump in the sequence. Endpoints are not repeated, or each turn would stutter. The MP4 has a scrubber, so it just plays forward.

**It is not a GIF because GIF is far larger.** Measured on a 128-frame ping-pong at 480 short edge:

| Format | Size |
| --- | --- |
| MP4 H.264 | 0.24 MB |
| **Animated WebP** | **1.75 MB** |
| GIF, 2-pass palette | 10.29 MB |
| GIF, naive palette | 23.18 MB |

GIF also caps at 256 colours, which bands visibly on photographic content.

### Why PNG and not a video

H.264 is lossy in ways that matter when frames feed further processing. 4:2:0 chroma subsampling stores colour at half resolution on both axes, and DCT quantisation concentrates its error on fine high-frequency detail. Measured on a synthetic 65-frame sequence at CRF 18, mean absolute error was **7.39 on fine vertical lines against 2.45 elsewhere** — a 3x concentration on exactly the detail downstream compositing depends on.

Formats measured on the same sequence (65 frames, 1920x1280):

| Format | Size | Round-trips bit-exact? |
| --- | --- | --- |
| PNG sequence | 172 MB | reference |
| FFV1 (RGB) | 152 MB | yes — max error 0 |
| x264rgb `-qp 0` | 146 MB | yes — max error 0 |
| ProRes 4444 | 106 MB | **no** — max error 21/255 |
| H.264 4:2:0 CRF 18 | 1.1 MB | no — max error 197/255 |

Two results worth noting. ProRes 4444 is *not* mathematically lossless despite the name — "4444" describes chroma sampling, not fidelity. And inter-frame prediction buys almost nothing: x264rgb's 146 MB against FFV1's 152 MB is a 4% gain, because per-frame sensor noise is uncorrelated between frames and defeats temporal prediction.

### Why views are selected here and not downstream

Returning all 97 frames at working resolution is not viable: 97 lossless frames at 16 MP is roughly 1.6 GB, against about 200 MB for the 12 typically kept.

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

## Frame selection

Views are spaced evenly across the **concatenated** sequence. The first and last frames are pinned — those are real captures, and dropping either would discard genuine detail.

### Interior captures are not pinned

`frame2` and `frame3` are **not** guaranteed to appear among the returned views, and this is deliberate. It is the whole reason segments are concatenated before selection rather than selected from individually.

Pinning the interior captures would force the selector onto indices that are not evenly spaced from their neighbours, putting a visible hitch in the parallax sweep at exactly the positions where the customer's real captures sit. Even spacing across the whole run matters more than landing on any particular source frame. The interior captures are still *in* the sequence; they simply may not be among the views returned.

They sometimes are anyway — at `num_views=5` with three inputs the selection is 0, 16, 32, 48, 64, and index 32 is `frame2` exactly. That is a coincidence of the arithmetic, not a guarantee, and nothing should depend on it.

### Do not lower `times_to_interpolate` to "save work"

Interpolating 65 or 97 frames and keeping 12 looks wasteful. It is not.

Selected views must be evenly spaced, or the parallax step between consecutive prints varies. Neither 11 nor 9 intervals divides any power of two, so exact spacing is impossible at **any** depth for 12 or 10 views — the only question is how uneven. Measured spread of gap sizes:

| | 3 inputs (2 segments) | 4 inputs (3 segments) |
| --- | --- | --- |
| `t=4`, 12 views | 33 frames, 34% spread | 49 frames, 23% spread |
| **`t=5`, 12 views** | **65 frames, 17% spread** | **97 frames, 12% spread** |
| `t=6`, 12 views | 129 frames, 9% spread | 193 frames, 6% spread |
| `t=5`, 10 views | 65 frames, 14% spread | 97 frames, 9% spread |
| `t=5`, 5 views | 65 frames, 0% spread | 97 frames, 0% spread |

Depth is **fixed at 5 for both input counts**. It does not need to vary with the number of inputs: a fourth frame adds a third segment, which raises the frame count from 65 to 97 and tightens the spread on its own. Five is the last depth whose 12-view spread stays under 20% at the three-input worst case while remaining at 1x GPU cost per segment.

Five views divide evenly at every depth and input count, so the 300 PPI product is always exactly spaced. The 720 and 600 PPI products are the constrained cases.

`times_to_interpolate` is exposed so it can be varied in the playground without a rebuild, not because production should change it.

### This is where realistec-multi diverges from realistec-stereo

The reason is **segment count, not a different judgement about quality.** Stereo has a single segment, so it needs depth 6 to reach 65 frames. This model reaches 65 at depth 5 because it has two segments, and 97 with three.

If you are porting a comment or a default from the stereo model, check which of those it was reasoning about.

## Block subdivision sizing

For a `block_height x block_width` grid, each patch is roughly `(input_height / block_height) x (input_width / block_width)` pixels. FILM was designed for inputs around 1920x1080, so size the grid to keep each patch in that ballpark:

| Input resolution | Recommended grid | Patch size |
| --- | --- | --- |
| 1920x1080 (1080p) | 1x1 | 1920x1080 |
| 2560x1440 (1440p) | 2x2 | 1280x720 |
| 3840x2160 (4K) | 2x2 | 1920x1080 |
| 7680x4320 (8K) | 4x4 | 1920x1080 |

Each patch adds prediction time roughly linearly, so a 2x2 grid takes about 4x as long as 1x1 — **per segment**, so a four-frame job at 2x2 is roughly 12 patch-pair passes' worth of work.

The caller computes these as `ceil(frame_height / 1920)` and `ceil(frame_width / 1920)`.

## Memory behaviour

Selected frames are written to disk as they are yielded and released immediately. **Do not reintroduce `list(self._recursive_interpolate(...))`.**

An early version of the stereo model built the full frame list in memory and then a uint8 copy alongside it, holding both at once:

| Input size | Peak host RAM (v1) | Peak host RAM (v2) |
| --- | --- | --- |
| 1920x1280 | 2.2 GB | ~0.5 GB |
| 3600x3600 | 11.8 GB | ~1.2 GB |
| 4000x4000 | 14.5 GB | ~1.4 GB |
| 5464x5464 | 27.1 GB | ~2.5 GB |

The v1 path could not complete a full-resolution job at all.

**This matters more here than it did there.** Four inputs produce 97 frames rather than 65, so anything scaling with total frame count costs half again as much. The one list still held at full length is `preview_frames`, at preview resolution rather than working resolution — 97 frames of 1620x1080x3 is about 500 MB. That is the deliberate ceiling on this loop, and it is why the preview is built at `preview_short_edge` rather than downscaled at the end.

## How it works

The model is FILM's stock pretrained `Style` SavedModel (the same one Google publishes on TF Hub as [`film/1`](https://www.kaggle.com/models/google/film)), bundled into the image at build time rather than fetched at runtime, which keeps builds reproducible and avoids a model pull on every scale-from-zero.

`_pad_to_align` and the call structure of `interpolate()` / `__call__()` are copied verbatim from Google Research's [`eval/interpolator.py`](https://github.com/google-research/frame-interpolation/blob/main/eval/interpolator.py). The Cog wrapper logic around them is original.

`image_to_patches` / `patches_to_image` began as Google's code but were **rewritten**. The originals split along an axis whose length is one patch's *pixel count* rather than the patch count — 2,300,881 tensors for a 1919x2399 input at a 2x1 grid — so any input large enough to need block subdivision ran indefinitely without erroring. Replaced with reshape + transpose, verified byte-identical to the original.

Preview frames are forced to even pixel dimensions for the MP4 on every path, including when no downscaling occurs. libx264 with `yuv420p` rejects odd dimensions, and that would otherwise surface as an ffmpeg failure at the very end of a long prediction. The WebP has no such constraint, so it is not rounded there.

The ping-pong tail costs no extra memory: it is the same frames indexed in reverse.

## Build environment

`cog.yaml` is unchanged from `realistec-stereo` apart from its comments — the multi-frame work is entirely in `predict.py` and adds no dependencies, using only `itertools.chain` from the stdlib.

Two things to know:

- It uses `python_requirements`, not the deprecated `python_packages`, so modern cog versions actually install the dependencies.
- The CUDA version is intentionally not pinned; cog auto-detects it for `tensorflow 2.15.0` from its compatibility matrix.

**No cog version is pinned either**, which is not a neutral choice: each push builds against whatever cog is current, and a runtime change between pushes can alter the published schema without any source change. That is exactly what broke `frame4` on the depth model. If a rebuild ever behaves differently from the deployed image for no apparent reason, compare the two versions' `openapi_schema` before looking anywhere else.

`cog.yaml` currently declares `predict: "predict.py:Predictor"`. Cog has since renamed the concept — `BaseRunner`, `Runner`, `run()`, and a `run:` key — and warns when it loads the legacy names. The old names still work, so this is a warning rather than an error, but it is a pending migration.

## Changes in this branch (`phase1-multi-frame`)

**Breaking.** The input contract changed from two frames to three-or-four.

- `frame3` added as required; `frame4` added as optional.
- `times_to_interpolate` default lowered from 6 to 5. The frame count is unchanged at 65 for three inputs, because two segments at depth 5 equal one segment at depth 6.
- Frame selection now spans the concatenated sequence and does not pin interior captures.
- `frame4` declared `Optional[Path]` so that three-frame predictions survive a rebuild.

## License

The Cog wrapper code in `predict.py` is released under the Apache License 2.0, the same license as the upstream FILM model. The bundled FILM SavedModel is copyright 2022 Google LLC, also Apache 2.0 licensed.

## Citation

```
@inproceedings{reda2022film,
 title = {FILM: Frame Interpolation for Large Motion},
 author = {Fitsum Reda and Janne Kontkanen and Eric Tabellion and Deqing Sun and Caroline Pantofaru and Brian Curless},
 booktitle = {European Conference on Computer Vision (ECCV)},
 year = {2022}
}
```
