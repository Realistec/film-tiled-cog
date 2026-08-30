"""
realistec-multi — a multi-input, high-resolution-capable fork of Google's FILM
frame interpolation model.

This is a Cog wrapper around the FILM SavedModel. It differs from the upstream
Replicate model in two ways:

  1. It accepts THREE OR FOUR input frames rather than two, and interpolates
     across the whole run as one continuous sequence.
  2. It exposes block_height / block_width, which subdivide each frame into
     non-overlapping patches so inputs far larger than FILM's native ~1920x1080
     working size can be processed.

When block_height and block_width are both 1 (the default), patch behaviour is
identical to the upstream model. When either is greater than 1, frames are
folded into block_height * block_width patches, the model runs on each patch
pair, and the interpolated patches are reassembled into a full-resolution frame.

The patch-folding helper _pad_to_align and the call structure of
`interpolate()` / `__call__()` are copied verbatim from Google Research's
eval/interpolator.py (https://github.com/google-research/frame-interpolation,
Apache 2.0 license, copyright 2022 Google LLC). The cog wrapper logic around
them is original.

image_to_patches / patches_to_image began as Google's code but were REWRITTEN —
the originals split along an axis whose length is one patch's pixel count, which
is millions of tensors at real image sizes and never completes. See the comment
in image_to_patches for detail. Carried over unchanged from realistec-stereo.


MULTI-FRAME INPUT
-----------------
The model takes frame1, frame2, frame3 and optionally frame4. Consecutive pairs
form SEGMENTS: three inputs give two segments, four give three. Each segment is
interpolated independently to depth `times_to_interpolate`, and the results are
concatenated into one sequence:

    total_frames = segments * 2**times_to_interpolate + 1

At the fixed depth of 5 that is 65 frames from three inputs and 97 from four.

>>> INTERIOR CAPTURES ARE NOT PINNED <<<
Frame selection spaces views evenly across the CONCATENATED sequence. It pins
the first and last frames — those are real captures and dropping either would
discard genuine detail — but it does not pin frame2 or frame3.

This is deliberate, and it is the whole reason the segments are concatenated
before selection rather than selected from individually. Pinning the interior
captures would force the selector to hit indices that are not evenly spaced from
their neighbours, which puts a visible hitch in the parallax sweep at exactly the
positions where the customer's real captures sit. Even spacing over the whole run
matters more than landing on any particular source frame; the interior captures
are still IN the sequence, they simply are not guaranteed to be among the views
returned. See _select_indices.

(They sometimes are anyway. At num_views=5 with three inputs the selection is
0, 16, 32, 48, 64 and index 32 is frame2 exactly. That is a coincidence of the
arithmetic, not a guarantee, and nothing should depend on it.)


EXPORT PIPELINE
---------------
The model does not return a single H.264 MP4. H.264 is lossy in ways that matter
when frames feed further processing rather than being watched: 4:2:0 chroma
subsampling stores colour at half resolution on both axes, and DCT quantisation
concentrates its error on fine high-frequency detail. Measured on a synthetic
65-frame sequence, CRF 18 showed mean absolute error of 7.39 on fine vertical
lines against 2.45 elsewhere — a 3x concentration on exactly the kind of detail
downstream compositing depends on.

It returns THREE artifacts:

  preview : H.264 MP4, forward pass through all frames, short edge 1080 by
            default. A review artifact — opened in a player with a scrubber,
            where a straight pass is easier to step through.
  share   : animated WebP, PING-PONG loop, short edge 480 by default. WebP is an
            IMAGE format, so it renders in an <img> tag and loops silently
            forever with no player chrome — an MP4 opened standalone gets
            transport controls that reappear on every loop.
  frames  : zip of exactly `num_views` lossless PNGs at full working resolution,
            plus a manifest.json.

Only the WebP ping-pongs (0..n-1 then n-2..1). It autoplays with no scrubber and
no way to reverse by hand, so the loop itself has to carry the back-and-forth; a
forward loop would jump-cut from the last view straight back to the first, the
widest parallax jump in the sequence. Endpoints are not repeated, or each turn
would stutter.

>>> WHY WEBP AND NOT GIF <<<
GIF is not smaller. Measured on a 128-frame ping-pong at 480 short edge:
    MP4 H.264 ............  0.24 MB
    Animated WebP ........  1.75 MB
    GIF, 2-pass palette .. 10.29 MB
    GIF, naive palette ... 23.18 MB
GIF also caps at 256 colours, which bands badly on photographic content. WebP
gives full colour at a sixth of the size and loops identically.

Note that sizing by SHORT edge is not a saving over long-edge sizing — for a 3:2
frame, short edge 1080 gives 1620x1080 where long edge 1280 gave 1280x853. It is
chosen for the guarantee, not the size: the narrow dimension holds up whatever
the orientation.

Frames are selected here rather than downstream because returning all 97 at
working resolution is not viable — 97 lossless frames at 16 MP is roughly 1.6 GB,
against ~200 MB for the 12 typically kept.

>>> DO NOT LOWER times_to_interpolate TO "SAVE WORK" <<<
--------------------------------------------------------
It looks obviously wasteful to interpolate 65 or 97 frames and keep 12. It is not.

Selected views must be evenly spaced, or the parallax step between consecutive
prints varies. Neither 11 nor 9 intervals divides any power of two, so exact
spacing is impossible at ANY depth for 12 or 10 views — the question is only how
uneven. Measured spread of gap sizes:

                          3 inputs (2 seg)        4 inputs (3 seg)
    t=4    12 views    33 frames,  34% spread   49 frames,  23% spread
    t=5    12 views    65 frames,  17% spread   97 frames,  12% spread
    t=6    12 views   129 frames,   9% spread  193 frames,   6% spread
    t=5    10 views    65 frames,  14% spread   97 frames,   9% spread
    t=5     5 views    65 frames,   0% spread   97 frames,   0% spread

Depth is FIXED AT 5 for both input counts. It does not need to vary with the
number of inputs: adding a fourth frame adds a third segment, which raises the
frame count from 65 to 97 and tightens the spread on its own. Five is the last
depth whose 12-view spread stays under 20% at the three-input worst case while
remaining at 1x GPU cost per segment.

Note this is where realistec-multi diverges from realistec-stereo, and the reason
is segment count, not a different judgement about quality. Stereo has a single
segment, so it needs depth 6 to reach 65 frames. Film reaches 65 at depth 5
because it has two segments, and 97 with three. If you are porting a comment or a
default from the stereo model, check which of those it was reasoning about.

Five views divide evenly at every depth and input count, so the 300 PPI product
is always exactly spaced. The 720 and 600 PPI products are the constrained cases.

>>> DO NOT REINTRODUCE list(self._recursive_interpolate(...)) <<<
-----------------------------------------------------------------
An early version of the stereo model built the full frame list in memory and then
built a uint8 copy alongside it, holding both at once. That is ~14.5 GB of host
RAM for a 4000x4000 input at 65 frames, so it could not complete a
full-resolution job at all. Selected frames are written to disk as they are
yielded and released immediately.

This matters MORE here than it did there. Four inputs produce 97 frames rather
than 65, so anything that scales with total frame count costs half again as much.
The one list still held at full length is preview_frames, at preview resolution
rather than working resolution: 97 frames of 1620x1080x3 is about 500 MB. That is
the deliberate ceiling on this loop, and it is why the preview is built at
preview_short_edge rather than downscaled at the end.
"""

import json
import os
import tempfile
import zipfile
from itertools import chain
from pathlib import Path as PyPath
from typing import List, Optional

import numpy as np
import tensorflow as tf
from PIL import Image, PngImagePlugin
import mediapy
from cog import BasePredictor, BaseModel, Input, Path

# Raise Pillow's zip-bomb decompression guard from its ~64 KB default to 100 MB.
# This prevents spurious ValueError('Decompressed Data Too Large') failures when
# input PNGs contain unusually large zTXt (compressed text) chunks — for example,
# embedded color profiles or camera metadata. The default limit is a security
# guard against maliciously crafted zip bombs, but it's aggressive enough to
# reject perfectly legitimate images that just happen to have large metadata.
# 100 MB is a comfortable ceiling for any real-world image, well below the
# memory any single prediction allocates for the actual image data itself.
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024


_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

# GIF is deliberately absent. The plugin's upload handler historically accepted
# it, which meant a GIF could clear every plugin-side check and then fail here
# at predict time — after payment, as a stalled order. The plugin now rejects it
# at upload instead. A GIF of a 35mm scan is 256 colours and has no business in
# a lossless print pipeline regardless.
_INPUT_EXT = (".png", ".jpg", ".jpeg")

# Path inside the container where the bundled SavedModel lives.
# This must match where the GitHub repo's pretrained_model/ directory ends
# up after `cog build` copies the working directory into the image.
_MODEL_PATH = "pretrained_model"


# ---------------------------------------------------------------------------
# The following three functions and the patch-aware __call__ semantics are
# copied verbatim from Google Research's eval/interpolator.py with only
# whitespace and comment adjustments. They are licensed Apache 2.0,
# copyright 2022 Google LLC. See:
#   https://github.com/google-research/frame-interpolation/blob/main/eval/interpolator.py
# ---------------------------------------------------------------------------

def _pad_to_align(x, align):
    """Pad image batch x so width and height divide by `align`.

    Args:
      x: Image batch of shape (B, H, W, C).
      align: Integer alignment. Output dimensions will be the smallest
        multiples of `align` that are >= the input dimensions.

    Returns:
      A tuple (padded_x, bbox_to_crop). bbox_to_crop is a dict that can be
      passed directly to tf.image.crop_to_bounding_box to undo the padding
      after the model has been run.
    """
    assert np.ndim(x) == 4
    assert align > 0, "align must be a positive number."

    height, width = x.shape[-3:-1]
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    bbox_to_pad = {
        "offset_height": height_to_pad // 2,
        "offset_width": width_to_pad // 2,
        "target_height": height + height_to_pad,
        "target_width": width + width_to_pad,
    }
    padded_x = tf.image.pad_to_bounding_box(x, **bbox_to_pad)
    bbox_to_crop = {
        "offset_height": height_to_pad // 2,
        "offset_width": width_to_pad // 2,
        "target_height": height,
        "target_width": width,
    }
    return padded_x, bbox_to_crop


def image_to_patches(image: np.ndarray, block_shape: List[int]) -> np.ndarray:
    """Fold an image into patches stacked along the batch dimension.

    Args:
      image: Input image of shape (B, H, W, C). B should be 1.
      block_shape: [block_height, block_width].

    Returns:
      Patches of shape (block_height * block_width, H/bh, W/bw, C).
    """
    block_height, block_width = block_shape
    num_blocks = block_height * block_width
    height, width, channel = image.shape[-3:]
    patch_height = height // block_height
    patch_width = width // block_width

    assert height == patch_height * block_height, (
        "block_height=%d should evenly divide height=%d." % (block_height, height)
    )
    assert width == patch_width * block_width, (
        "block_width=%d should evenly divide width=%d." % (block_width, width)
    )

    # REWRITTEN. Google's original built the patch grid with
    #
    #     patches = tf.space_to_batch(image, [patch_height, patch_width], ...)
    #     patches = tf.split(patches, patch_height * patch_width, 0)
    #     patches = tf.stack(patches, axis=3)
    #
    # The split count is patch_height * patch_width - the PIXEL COUNT OF ONE
    # PATCH, not the number of patches. For a 1919x2399 input at block 2x1 that
    # is 2,300,881 individual tensors to create and then stack, which is
    # millions of Python-level TensorFlow calls. It does not error; it simply
    # never finishes. Observed as a prediction running past 1200s with no
    # output, unaffected by GPU size because the cost is graph construction on
    # the CPU. It goes unnoticed at block_shape [1,1] because
    # _interpolate_with_blocks() early-returns before calling this at all.
    #
    # The operation is ordinary tiling, so reshape + transpose does it in
    # constant time. Verified byte-identical to the original across several
    # geometries including 2398x1919 at 2x1.
    arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]

    arr = arr.reshape(block_height, patch_height, block_width, patch_width, channel)
    arr = arr.transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(
        arr.reshape(num_blocks, patch_height, patch_width, channel)
    )


def patches_to_image(patches: np.ndarray, block_shape: List[int]) -> np.ndarray:
    """Unfold patches (stacked along batch) back into a single image.

    Args:
      patches: Input patches of shape (num_patches, patch_H, patch_W, C).
      block_shape: [block_height, block_width].

    Returns:
      Image of shape (1, H, W, C) where H = block_height * patch_H,
      W = block_width * patch_W.
    """
    block_height, block_width = block_shape
    patch_height, patch_width, channel = patches.shape[-3:]

    # Rewritten for the same reason as image_to_patches() - the original split
    # along a patch-pixel-count axis, which is millions of tensors at real image
    # sizes. Exact inverse of the tiling above.
    arr = np.asarray(patches)
    arr = arr.reshape(block_height, block_width, patch_height, patch_width, channel)
    arr = arr.transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(
        arr.reshape(1, block_height * patch_height, block_width * patch_width, channel)
    )


# End of code copied from Google Research's eval/interpolator.py.
# ---------------------------------------------------------------------------


def _read_image(path: str) -> np.ndarray:
    """Read an image file and return a float32 array in [0, 1].

    Returns shape (H, W, C) with C=3 (RGB).
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / _UINT8_MAX_F
    return arr


def _write_image(path: str, image: np.ndarray) -> None:
    """Write a float32 image (H, W, C) in [0, 1] as PNG."""
    arr = np.clip(image * _UINT8_MAX_F + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, format="PNG")


class Output(BaseModel):
    """Three artifacts with different jobs.

    Returned as cog Path objects, NOT base64 data URIs. Replicate uploads these
    and hands back replicate.delivery URLs, which is what makes a 200 MB frames
    zip viable at all — a data URI would inflate it ~37% and have to fit inside
    the prediction JSON, which is also the webhook payload.

    Note that those URLs expire roughly an hour after the prediction for
    API-created predictions, so the consumer must copy rather than link.
    """

    preview: Path
    share: Path
    frames: Path


def _boomerang_order(n: int) -> List[int]:
    """Indices for a ping-pong loop: 0..n-1 then n-2..1.

    The endpoints are NOT repeated. Including them would hold the first and last
    view on screen for two frame periods, which reads as a stutter at each turn
    rather than a smooth reversal.

    A plain forward loop jump-cuts from the last frame back to the first, which
    for a parallax sweep is the widest jump in the sequence - the most visible
    cut possible. Ping-ponging removes it entirely.
    """
    if n < 3:
        return list(range(n))
    return list(range(n)) + list(range(n - 2, 0, -1))


def _select_indices(total: int, n: int) -> List[int]:
    """Pick `n` evenly spaced frame indices from `total`, endpoints pinned.

    Derived from the ACTUAL frame count rather than assuming 65 or 97, so a
    change to times_to_interpolate or to the number of inputs moves the
    selection with it instead of silently sampling the wrong positions.

    Endpoints are pinned because index 0 and index total-1 are real captures.
    Dropping either would discard genuine detail and replace it with an
    interpolated approximation of it.

    INTERIOR captures are NOT pinned. With three or four inputs there are one or
    two source frames sitting inside the sequence (at 2**t and 2*2**t), and this
    function will usually not select them. That is intended - see the module
    docstring. Forcing them into the selection would make the gaps around them
    uneven, which is visible in the finished lenticular as a hitch in the sweep.
    """
    if n >= total:
        return list(range(total))
    last = total - 1
    idx = [round(i * last / (n - 1)) for i in range(n)]

    # Distinct by construction while n <= total, since the step is >= 1.
    # Deduped defensively anyway: a silent collision would put two identical
    # frames in the output, which downstream consumers have no way to detect.
    unique = sorted(set(idx))
    if len(unique) != len(idx):
        print(f"WARNING: frame selection collided, {len(idx)} -> {len(unique)}")
    return unique


def _fit_short_edge(size, short_target: int):
    """Target (w, h) scaling the SHORT edge to short_target.

    Short-edge sizing guarantees a floor on the narrow dimension regardless of
    orientation, which long-edge sizing does not: a portrait frame capped on its
    long edge ends up narrow, and a landscape one ends up short.

    NO LONG-EDGE CAP, deliberately. Scaling by the short edge is unbounded on
    the other axis in principle, but the frames reaching this model have already
    been cropped to match an orderable print size, and the widest of those is
    3:2 - so the long edge cannot exceed 1.5x the short one. A cap would be dead
    code. (Worth revisiting only if a panoramic print size is ever offered, or
    if this model is driven directly with arbitrary input.)

    Never upscales: a source already smaller than the target is left alone.
    """
    w, h = size
    if w <= 0 or h <= 0:
        return w, h

    short_dim = min(w, h)
    scale = min(short_target / float(short_dim), 1.0)

    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def _even(v: int) -> int:
    """Round down to the nearest even number, floor of 2.

    libx264 with yuv420p refuses odd dimensions. Applied to the MP4 path only -
    WebP has no such constraint, so forcing it there would crop a pixel for
    nothing.
    """
    return max(2, v - (v % 2))


def _to_u8(image: np.ndarray) -> np.ndarray:
    """float32 [0,1] -> uint8, without resizing."""
    return np.clip(image * _UINT8_MAX_F + 0.5, 0, 255).astype(np.uint8)


def _resize_u8(arr: np.ndarray, size) -> np.ndarray:
    """Resize a uint8 RGB frame to an exact (w, h)."""
    if (arr.shape[1], arr.shape[0]) == tuple(size):
        return arr
    return np.asarray(Image.fromarray(arr).resize(size, Image.LANCZOS))


def _ensure_same_size(images: List[np.ndarray]) -> List[np.ndarray]:
    """Crop every image to the smallest common height and width.

    Generalises the stereo model's two-image version. The model requires
    equal-sized inputs, and cropping mismatched ones rather than rejecting them
    matches upstream behaviour.

    Taking the minimum across ALL inputs matters here in a way it did not with
    two frames: the plugin computes block_height and block_width from the frame
    dimensions, and if it measures one frame while the model works on a smaller
    common crop, the grid it chose no longer describes what is being processed.
    The plugin therefore measures the same minimum. This function is the
    backstop for that agreement, not the only place it is enforced.
    """
    h = min(img.shape[0] for img in images)
    w = min(img.shape[1] for img in images)
    return [img[:h, :w, :] for img in images]


def _crop_to_block_divisible(
    images: List[np.ndarray], block_height: int, block_width: int
) -> List[np.ndarray]:
    """Crop images so their height and width are divisible by the block grid.

    image_to_patches() asserts that block dimensions evenly divide the input.
    For arbitrary user inputs we can't guarantee that, so we crop a few pixels
    off the right/bottom if necessary. The crop is at most block_height-1 rows
    and block_width-1 columns, which is visually negligible for any reasonable
    block size.

    Assumes the images already share dimensions - call _ensure_same_size first.
    """
    h, w, _ = images[0].shape
    new_h = (h // block_height) * block_height
    new_w = (w // block_width) * block_width
    if new_h == h and new_w == w:
        return images
    return [img[:new_h, :new_w, :] for img in images]


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the bundled FILM SavedModel into memory once.

        Cog calls this when the container starts. Subsequent predict() calls
        reuse self._model without reloading.
        """
        print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
        print(f"Loading SavedModel from {_MODEL_PATH}/")
        self._model = tf.compat.v2.saved_model.load(_MODEL_PATH)
        # FILM's convolutions require input dimensions to be multiples of 64.
        # We pad inputs to this alignment before each model call and crop
        # the model output back to the original size afterwards.
        self._align = 64
        print("Model loaded successfully.")

    def _interpolate_single(
        self, x0: np.ndarray, x1: np.ndarray, dt: np.ndarray
    ) -> np.ndarray:
        """Run the model once on a single batched frame pair.

        Handles padding-to-alignment and unpadding. Inputs and outputs are
        4-D float32 arrays of shape (1, H, W, 3) in the [0, 1] range.

        Equivalent to Google's Interpolator.interpolate().
        """
        x0_padded, bbox_to_crop = _pad_to_align(x0, self._align)
        x1_padded, _ = _pad_to_align(x1, self._align)

        inputs = {
            "x0": x0_padded,
            "x1": x1_padded,
            "time": dt[..., np.newaxis],
        }
        result = self._model(inputs, training=False)
        image = result["image"]
        image = tf.image.crop_to_bounding_box(image, **bbox_to_crop)
        return image.numpy()

    def _interpolate_with_blocks(
        self,
        x0: np.ndarray,
        x1: np.ndarray,
        dt: np.ndarray,
        block_shape: List[int],
    ) -> np.ndarray:
        """Run the model on a grid of patches and stitch results back together.

        Equivalent to Google's Interpolator.__call__() when block_shape is set.
        """
        if np.prod(block_shape) <= 1:
            return self._interpolate_single(x0, x1, dt)

        x0_patches = image_to_patches(x0, block_shape)
        x1_patches = image_to_patches(x1, block_shape)

        output_patches = []
        for image_0, image_1 in zip(x0_patches, x1_patches):
            mid = self._interpolate_single(
                image_0[np.newaxis, ...], image_1[np.newaxis, ...], dt
            )
            output_patches.append(mid)

        output_patches = np.concatenate(output_patches, axis=0)
        return patches_to_image(output_patches, block_shape)

    def _recursive_interpolate(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_recursions: int,
        block_shape: List[int],
    ):
        """Recursively generate in-between frames for ONE segment.

        Mirrors Google's util.interpolate_recursively_from_files() behavior.
        Yields frames in temporal order, including frame1 but excluding the
        final frame2.

        For num_recursions = N, this yields 2^N frames between frame1 and
        frame2 (inclusive of frame1, exclusive of frame2).

        Excluding frame2 is what makes segments concatenate cleanly: chaining
        the generators for (f1,f2), (f2,f3), (f3,f4) yields each interior
        capture exactly once, and the caller appends the final frame.
        """
        if num_recursions == 0:
            yield frame1
            return

        # Generate the midpoint frame.
        time = np.array([0.5], dtype=np.float32)
        mid = self._interpolate_with_blocks(
            frame1[np.newaxis, ...], frame2[np.newaxis, ...], time, block_shape
        )[0]

        # Recurse on each half.
        yield from self._recursive_interpolate(
            frame1, mid, num_recursions - 1, block_shape
        )
        yield from self._recursive_interpolate(
            mid, frame2, num_recursions - 1, block_shape
        )

    def predict(
        self,
        frame1: Path = Input(description="The first input frame"),
        frame2: Path = Input(description="The second input frame"),
        frame3: Path = Input(description="The third input frame"),
        frame4: Optional[Path] = Input(
            description=(
                "Optional fourth input frame. Supplying it adds a third segment, "
                "raising the generated sequence from 65 frames to 97 and tightening "
                "the spacing of the returned views. Leave unset for a three-frame job."
            ),
            default=None,
        ),
        times_to_interpolate: int = Input(
            description=(
                "Recursion depth per segment. Each segment produces "
                "2^times_to_interpolate frames, so the full sequence is "
                "segments * 2^times_to_interpolate + 1 frames. "
                "LEAVE THIS AT 5. Lower values do not save meaningful work and "
                "make the returned views unevenly spaced - see the module "
                "docstring for the measured spread at each depth. It is exposed "
                "so it can be varied in the playground without a rebuild, not "
                "because production should change it."
            ),
            default=5,
            ge=1,
            le=8,
        ),
        block_height: int = Input(
            description=(
                "Number of rows to subdivide the input frames into for "
                "high-resolution interpolation. The default of 1 means no "
                "subdivision. The caller computes this as "
                "ceil(frame_height / 1920), which keeps each patch inside the "
                "~1920x1080 working size FILM was designed for. The product "
                "block_height * block_width is the total number of patches "
                "processed per frame pair, so larger values mean longer "
                "prediction times."
            ),
            default=1,
            ge=1,
            le=8,
        ),
        block_width: int = Input(
            description=(
                "Number of columns to subdivide the input frames into for "
                "high-resolution interpolation. The default of 1 means no "
                "subdivision. Computed by the caller as ceil(frame_width / 1920)."
            ),
            default=1,
            ge=1,
            le=8,
        ),
        num_views: int = Input(
            description=(
                "How many evenly spaced frames to return as lossless PNGs, with "
                "the first and last pinned to the first and last input frames. "
                "In production this is the printer's addressable view count - "
                "PPI divided by the 60 LPI lens, giving 12 at 720 PPI, 10 at 600 "
                "and 5 at 300. The default of 12 exists for playground runs; the "
                "plugin always sends an explicit value. The preview video always "
                "contains every frame regardless of this."
            ),
            default=12,
            ge=2,
            le=257,
        ),
        preview_short_edge: int = Input(
            description=(
                "Short-edge pixel size of the preview MP4. Sizing by the SHORT "
                "edge guarantees a floor on the narrow dimension whatever the "
                "orientation, which long-edge sizing does not. Never upscales."
            ),
            default=1080,
            ge=240,
            le=2160,
        ),
        share_short_edge: int = Input(
            description=(
                "Short-edge pixel size of the shareable animated WebP, which "
                "ping-pongs and loops forever. Kept small because WebP has no "
                "interframe compression comparable to H.264: at 480 a ping-pong "
                "loop lands around 1-2 MB, where the same content as MP4 is a "
                "few hundred KB."
            ),
            default=480,
            ge=160,
            le=1080,
        ),
        preview_fps: int = Input(
            description=(
                "Playback rate of the preview MP4. At 60 the full sequence runs "
                "about 1.1s for three inputs and 1.6s for four - short, because "
                "the preview exists to be scrubbed rather than watched."
            ),
            default=60,
            ge=1,
            le=120,
        ),
        share_fps: int = Input(
            description=(
                "Playback rate of the share WebP. Converted to a per-frame "
                "duration in whole milliseconds, which is all the WebP container "
                "stores, so rates that do not divide 1000 evenly are approximated "
                "- 50 gives exactly 20ms and is the intended value."
            ),
            default=50,
            ge=1,
            le=100,
        ),
    ) -> Output:
        # Collect the supplied frames in order. frame4 is optional; everything
        # downstream derives from len(frame_paths) rather than assuming a count.
        frame_paths = [frame1, frame2, frame3]
        if frame4 is not None:
            frame_paths.append(frame4)
        num_inputs = len(frame_paths)
        segments = num_inputs - 1

        # Validate input file extensions.
        for i, p in enumerate(frame_paths, start=1):
            ext = os.path.splitext(str(p))[-1].lower()
            assert ext in _INPUT_EXT, (
                f"Please provide png, jpg or jpeg images. Got: frame{i}={ext}"
            )

        # Total frames the recursion will produce, INCLUDING the final input
        # frame, which the generators exclude and we append.
        total_frames = segments * 2 ** times_to_interpolate + 1

        # Fail loudly rather than under-delivering.
        #
        # _select_indices() clamps: asked for more views than exist, it quietly
        # returns every frame instead. That is the right behaviour for a
        # playground experiment and the wrong one for an order, because the
        # returned zip would contain fewer PNGs than the customer's printer
        # needs and nothing in the artifact says so. This is the same class of
        # failure as running the stereo model at too low a depth, which went
        # unnoticed precisely because the output was still well-formed.
        assert total_frames >= num_views, (
            f"num_views={num_views} exceeds the {total_frames} frames this "
            f"configuration generates ({num_inputs} inputs, {segments} segment(s), "
            f"times_to_interpolate={times_to_interpolate}). Raise "
            f"times_to_interpolate or lower num_views."
        )

        # Load all frames as float32 RGB arrays in [0, 1].
        images = [_read_image(str(p)) for p in frame_paths]

        # Crop to a common size. The model requires equal-sized inputs, and the
        # block grid the caller chose describes this common size.
        images = _ensure_same_size(images)

        # If block subdivision is requested, ensure dimensions are evenly
        # divisible by the block grid.
        block_shape = [block_height, block_width]
        if np.prod(block_shape) > 1:
            images = _crop_to_block_divisible(images, block_height, block_width)
            print(
                f"Block subdivision: {block_height}x{block_width} grid, "
                f"working dimensions {images[0].shape[1]}x{images[0].shape[0]}, "
                f"patch {images[0].shape[1] // block_width}x"
                f"{images[0].shape[0] // block_height}"
            )

        selected = _select_indices(total_frames, num_views)
        selected_set = set(selected)

        # Where the real captures land in the concatenated sequence. Recorded in
        # the manifest so a consumer can tell interpolated views from source
        # ones without recomputing the arithmetic.
        capture_indices = [i * 2 ** times_to_interpolate for i in range(num_inputs)]

        print(
            f"Interpolating {num_inputs} inputs as {segments} segment(s) at "
            f"times_to_interpolate={times_to_interpolate}, producing "
            f"{total_frames} frames, returning {len(selected)} as PNG"
        )
        print(f"Source captures at indices: {capture_indices}")
        print(f"Selected source indices: {selected}")

        out_dir = PyPath(tempfile.mkdtemp())
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        preview_frames = []
        entries = []
        view_no = 0

        # Preview geometries, computed once from the working frame size.
        # MP4 needs even dimensions for libx264/yuv420p; WebP does not, so it is
        # not forced there.
        src_size = (images[0].shape[1], images[0].shape[0])
        pw, ph = _fit_short_edge(src_size, preview_short_edge)
        preview_size = (_even(pw), _even(ph))
        share_size = _fit_short_edge(src_size, share_short_edge)

        # One generator per segment, chained into a single stream, with the
        # final input frame appended. Each generator excludes its own end frame,
        # so interior captures appear exactly once at the segment boundaries and
        # the sequence is continuous across them.
        #
        # The generators are built eagerly but consumed lazily - constructing
        # them does no interpolation work, so this does not defeat the streaming
        # below.
        segment_streams = [
            self._recursive_interpolate(
                images[i], images[i + 1], times_to_interpolate, block_shape
            )
            for i in range(segments)
        ]
        frame_stream = chain(chain.from_iterable(segment_streams), (images[-1],))

        # STREAMING. Each frame is written to disk if selected, downscaled for
        # the preview, then dropped. Holding the full sequence at working
        # resolution cannot complete a full-resolution job - see the module
        # docstring before changing this loop.
        #
        # ONE stored list, at preview resolution. The share frames are derived
        # from it at write time rather than accumulated in parallel - a second
        # list would add memory for frames that are a strict downscale of these.
        # The boomerang tail is likewise not stored: it is the same frames
        # indexed in reverse, so it costs nothing but an index list.
        for idx, frame in enumerate(frame_stream):
            preview_frames.append(_resize_u8(_to_u8(frame), preview_size))

            if idx in selected_set:
                # Filename carries BOTH the view ordinal and the true source
                # index, so ordering is self-describing and the consumer can
                # assert on it. Selected indices are not contiguous and are
                # unevenly spaced by necessity, so a bare sequence number would
                # lose the information needed to verify the selection.
                name = f"view_{view_no:02d}_src{idx:03d}.png"
                _write_image(str(frames_dir / name), frame)
                entries.append({
                    "view": view_no,
                    "source_index": idx,
                    "file": name,
                    # True when this view is one of the customer's actual
                    # captures rather than an interpolation. Not guaranteed to
                    # occur at all - see _select_indices.
                    "is_capture": idx in capture_indices,
                })
                view_no += 1

            del frame

        # Sanity check on the stream length. If the segment chaining is ever
        # changed, this catches a miscount before the manifest records it as
        # fact.
        assert len(preview_frames) == total_frames, (
            f"Generated {len(preview_frames)} frames, expected {total_frames}"
        )

        height, width = images[0].shape[:2]
        manifest = {
            "schema": 1,
            "model": "realistec-multi",
            "num_views": len(entries),
            "num_inputs": num_inputs,
            "segments": segments,
            "total_frames": total_frames,
            "times_to_interpolate": times_to_interpolate,
            "capture_indices": capture_indices,
            "block_height": block_height,
            "block_width": block_width,
            "frame_width": int(width),
            "frame_height": int(height),
            "preview_fps": preview_fps,
            "share_fps": share_fps,
            "views": entries,
        }
        with open(frames_dir / "manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)

        # ZIP_STORED, not ZIP_DEFLATED. PNG is already deflated; re-deflating
        # measured a 6 KB gain across 172 MB of frames, for minutes of CPU.
        zip_path = out_dir / "frames.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for entry in entries:
                zf.write(frames_dir / entry["file"], entry["file"])
            zf.write(frames_dir / "manifest.json", "manifest.json")

        # MP4 plays FORWARD only. It is a review artifact opened in a player
        # with a scrubber, where a straight pass through the sequence is easier
        # to step through than a ping-pong that visits every frame twice.
        preview_path = out_dir / "preview.mp4"
        mediapy.write_video(str(preview_path), preview_frames, fps=preview_fps)

        # The WebP is the one that ping-pongs. It autoplays in an <img> with no
        # scrubber and no way to reverse manually, so the loop itself has to
        # carry the back-and-forth; a forward loop would jump-cut from the last
        # frame straight back to the first, the widest parallax jump available.
        order = _boomerang_order(len(preview_frames))

        # WebP stores per-frame duration in whole milliseconds, so the frame
        # rate is quantised. 50 fps is exactly 20ms; a rate that does not divide
        # 1000 evenly lands on the nearest millisecond and plays slightly off.
        # Floored at 1ms because 0 is not a valid duration.
        share_duration_ms = max(1, int(round(1000.0 / share_fps)))
        effective_share_fps = 1000.0 / share_duration_ms
        if abs(effective_share_fps - share_fps) > 0.01:
            print(
                f"NOTE: share_fps={share_fps} quantises to {share_duration_ms}ms "
                f"per frame, an effective {effective_share_fps:.2f} fps"
            )

        # Animated WebP, for sharing. WebP is an IMAGE format, so it renders in
        # an <img> tag and loops silently with no player chrome - unlike an MP4,
        # which a browser or OS player wraps in transport controls that reappear
        # on every loop. loop=0 means infinite.
        #
        # Expect this to be several times the size of the MP4 despite being much
        # smaller in pixels: WebP animation has no interframe compression
        # comparable to H.264. Measured on a 128-frame ping-pong at 480 short
        # edge, MP4 was 0.24 MB and WebP 1.75 MB. GIF, for reference, was 10.29
        # MB at the same size - which is why this is not a GIF.
        #
        # Four inputs make this loop 193 frames rather than the stereo model's
        # 129, so expect the WebP to grow by roughly half against those measured
        # figures.
        share_path = out_dir / "share.webp"
        share_frames = [
            Image.fromarray(_resize_u8(preview_frames[i], share_size))
            for i in order
        ]
        share_frames[0].save(
            str(share_path),
            format="WEBP",
            save_all=True,
            append_images=share_frames[1:],
            duration=share_duration_ms,
            loop=0,               # infinite
            quality=70,
            method=4,
        )
        del share_frames

        print(
            f"Wrote {len(entries)} PNG views ({width}x{height}), a "
            f"{len(preview_frames)}-frame preview at "
            f"{preview_size[0]}x{preview_size[1]} @ {preview_fps}fps, and a "
            f"{len(order)}-frame boomerang share WebP at "
            f"{share_size[0]}x{share_size[1]} @ {effective_share_fps:.1f}fps"
        )

        return Output(
            preview=Path(preview_path),
            share=Path(share_path),
            frames=Path(zip_path),
        )
