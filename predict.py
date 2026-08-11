"""
realistec-stereo — a high-resolution-capable fork of Google's FILM frame
interpolation model.

This is a minimal Cog wrapper around the FILM SavedModel. It adds two
parameters that the upstream Replicate model does not expose:

  block_height : int   — subdivide input frames into this many rows
  block_width  : int   — subdivide input frames into this many columns

When both are 1 (the default), behavior is identical to the upstream model.
When either is greater than 1, the input frames are folded into
block_height * block_width non-overlapping patches, the model is run on
each patch pair, and the resulting interpolated patches are reassembled
into a single full-resolution output frame.

The patch-folding helpers (_pad_to_align, image_to_patches,
patches_to_image) and the call structure of `interpolate()` / `__call__()`
are copied verbatim from Google Research's eval/interpolator.py
(https://github.com/google-research/frame-interpolation, Apache 2.0
license, copyright 2022 Google LLC). The cog wrapper logic around them
is original.


EXPORT PIPELINE (v2)
--------------------
The model no longer returns a single H.264 MP4. H.264 is lossy in ways that
matter when frames feed further processing rather than being watched: 4:2:0
chroma subsampling stores colour at half resolution on both axes, and DCT
quantisation concentrates its error on fine high-frequency detail. Measured on
a synthetic 65-frame sequence, CRF 18 showed mean absolute error of 7.39 on
fine vertical lines against 2.45 elsewhere - a 3x concentration on exactly the
kind of detail that downstream compositing depends on.

It now returns TWO artifacts:

  preview : H.264 MP4, ALL frames, downscaled. A quick visual check, not a
            deliverable. Roughly 1 MB.
  frames  : zip of exactly `num_views` lossless PNGs at full working resolution,
            plus a manifest.json.

Frames are selected here rather than downstream because returning all 65 at
working resolution is not viable - 65 lossless frames at 16 MP is roughly
1.1 GB, against ~200 MB for the 12 typically kept.

>>> DO NOT LOWER times_to_interpolate TO "SAVE WORK" <<<
--------------------------------------------------------
It looks obviously wasteful to interpolate 65 frames and keep 12. It is not.

Selected frames must be evenly spaced in time, or the step between consecutive
outputs varies. Neither 11 nor 9 intervals divides any power of two, so exact
spacing is impossible at ANY depth - the question is only how uneven. Measured
spread of gap sizes when selecting 12 frames:

    times_to_interpolate=4  ->  17 frames  ->  gaps 1,2,1,2,...   69% spread
    times_to_interpolate=5  ->  33 frames  ->  gaps 3,3,3,2,...   34% spread
    times_to_interpolate=6  ->  65 frames  ->  gaps 6,6,5,6,...   17% spread
    times_to_interpolate=7  -> 129 frames  ->  gaps 12,11,12,...   9% spread

At depth 3 some gaps reach ZERO - two byte-identical frames in the output.
Depth 6 is the last one at 1x GPU cost and is the intended operating point.

>>> DO NOT REINTRODUCE list(self._recursive_interpolate(...)) <<<
-----------------------------------------------------------------
The v1 predict() built the full frame list in memory and then built a uint8
copy alongside it, holding both at once. That is ~14.5 GB of host RAM for a
4000x4000 input and ~11.8 GB for 3600x3600, so the v1 path could not complete
a full-resolution job at all. Selected frames are now written to disk as they
are yielded and released immediately; peak drops to ~1.4 GB.
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

    patch_size = patch_height * patch_width
    paddings = 2 * [[0, 0]]

    patches = tf.space_to_batch(image, [patch_height, patch_width], paddings)
    patches = tf.split(patches, patch_size, 0)
    patches = tf.stack(patches, axis=3)
    patches = tf.reshape(
        patches, [num_blocks, patch_height, patch_width, channel]
    )
    return patches.numpy()


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
    paddings = 2 * [[0, 0]]
    patch_height, patch_width, channel = patches.shape[-3:]
    patch_size = patch_height * patch_width

    patches = tf.reshape(
        patches, [1, block_height, block_width, patch_size, channel]
    )
    patches = tf.split(patches, patch_size, axis=3)
    patches = tf.stack(patches, axis=0)
    patches = tf.reshape(
        patches, [patch_size, block_height, block_width, channel]
    )
    image = tf.batch_to_space(patches, [patch_height, patch_width], paddings)
    return image.numpy()


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
    """Two artifacts with different jobs.

    Returned as cog Path objects, NOT base64 data URIs. Replicate uploads these
    and hands back replicate.delivery URLs, which is what makes a 200 MB frames
    zip viable at all — a data URI would inflate it ~37% and have to fit inside
    the prediction JSON, which is also the webhook payload.

    Note that those URLs expire roughly an hour after the prediction for
    API-created predictions, so the consumer must copy rather than link.
    """

    preview: Path
    frames: Path


def _select_indices(total: int, n: int) -> List[int]:
    """Pick `n` evenly spaced frame indices from `total`, endpoints pinned.

    Derived from the ACTUAL frame count rather than assuming 65. If
    times_to_interpolate is ever changed the selection follows it instead of
    silently sampling the wrong positions.

    Endpoints are pinned because index 0 and index total-1 are the two real
    source images. Dropping either would discard genuine captured detail and
    replace it with an interpolated approximation of it.
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


def _even(v: int) -> int:
    """Round down to the nearest even number, floor of 2.

    libx264 with yuv420p refuses odd dimensions. The downscale below is the
    only place a frame can acquire one, so it is clamped here rather than
    discovered as an ffmpeg failure at the very end of a long prediction.
    """
    return max(2, v - (v % 2))


def _downscale_u8(image: np.ndarray, long_edge: int) -> np.ndarray:
    """Convert a float32 [0,1] frame to uint8, downscaling to `long_edge`.

    Applied to EVERY frame for the preview video. Kept deliberately cheap: the
    preview is only a visual check, and holding 65 full-resolution frames in
    memory to build it would reinstate the exact memory problem this rewrite
    exists to remove.
    """
    arr = np.clip(image * _UINT8_MAX_F + 0.5, 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    if max(h, w) <= long_edge:
        # Still enforce even dimensions - the source can be odd-sized.
        if h % 2 == 0 and w % 2 == 0:
            return arr
        return np.asarray(
            Image.fromarray(arr).resize((_even(w), _even(h)), Image.LANCZOS)
        )
    scale = long_edge / float(max(h, w))
    return np.asarray(
        Image.fromarray(arr).resize(
            (_even(int(round(w * scale))), _even(int(round(h * scale)))),
            Image.LANCZOS,
        )
    )


def _ensure_same_size(img1: np.ndarray, img2: np.ndarray):
    """Crop both images to the smaller of their dimensions on each axis.

    Mirrors the behavior of the upstream Replicate predict.py, which
    crops mismatched inputs to a common size rather than rejecting them.
    Returns the (possibly cropped) (img1, img2) pair.
    """
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return img1[:h, :w, :], img2[:h, :w, :]


def _crop_to_block_divisible(
    img1: np.ndarray, img2: np.ndarray, block_height: int, block_width: int
):
    """Crop both images so their height and width are divisible by the block grid.

    image_to_patches() asserts that block dimensions evenly divide the input.
    For arbitrary user inputs we can't guarantee that, so we crop a few pixels
    off the right/bottom if necessary. The crop is at most block_height-1 rows
    and block_width-1 columns, which is visually negligible for any reasonable
    block size.
    """
    h, w, _ = img1.shape
    new_h = (h // block_height) * block_height
    new_w = (w // block_width) * block_width
    if new_h == h and new_w == w:
        return img1, img2
    return img1[:new_h, :new_w, :], img2[:new_h, :new_w, :]


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
        """Recursively generate in-between frames.

        Mirrors Google's util.interpolate_recursively_from_files() behavior.
        Yields frames in temporal order, including frame1 but excluding the
        final frame2 (the caller appends frame2 separately).

        For num_recursions = N, this yields 2^N frames between frame1 and
        frame2 (inclusive of frame1, exclusive of frame2).
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
        times_to_interpolate: int = Input(
            description=(
                "Controls the number of times the frame interpolator is invoked. "
                "If set to 1, the output will be the sub-frame at t=0.5 as a PNG. "
                "When set to > 1, the output will be the interpolation video with "
                "(2^times_to_interpolate + 1) frames at 30 fps."
            ),
            default=1,
            ge=1,
            le=8,
        ),
        block_height: int = Input(
            description=(
                "Number of rows to subdivide the input frames into for "
                "high-resolution interpolation. The default of 1 means no "
                "subdivision (identical to the upstream Replicate model). "
                "Use 2 for 4K (3840x2160) input. The product block_height * "
                "block_width is the total number of patches the model will "
                "process per frame, so larger values mean longer prediction times."
            ),
            default=1,
            ge=1,
            le=8,
        ),
        block_width: int = Input(
            description=(
                "Number of columns to subdivide the input frames into for "
                "high-resolution interpolation. The default of 1 means no "
                "subdivision (identical to the upstream Replicate model). "
                "Use 2 for 4K (3840x2160) input."
            ),
            default=1,
            ge=1,
            le=8,
        ),
        num_views: int = Input(
            description=(
                "How many evenly spaced frames to return as lossless PNGs, "
                "with the first and last pinned to the two input frames. "
                "Clamped to the number of frames actually generated. The "
                "preview video always contains every frame regardless of this "
                "value."
            ),
            default=12,
            ge=2,
            le=257,
        ),
        preview_long_edge: int = Input(
            description=(
                "Long-edge pixel size of the preview MP4. The preview is a "
                "quick visual check rather than a deliverable, so this stays "
                "small: 1280 keeps it near 1 MB for a 65-frame sequence. "
                "Never upscales, and does not affect the PNGs."
            ),
            default=1280,
            ge=240,
            le=3840,
        ),
    ) -> Output:
        # Validate input file extensions.
        ext1 = os.path.splitext(str(frame1))[-1].lower()
        ext2 = os.path.splitext(str(frame2))[-1].lower()
        assert ext1 in _INPUT_EXT and ext2 in _INPUT_EXT, (
            "Please provide png, jpg or jpeg images. "
            f"Got: frame1={ext1}, frame2={ext2}"
        )

        # Load both frames as float32 RGB arrays in [0, 1].
        img1 = _read_image(str(frame1))
        img2 = _read_image(str(frame2))

        # If the input frames have mismatched dimensions, crop both to the
        # common (smaller) size. This matches upstream behavior — the model
        # itself requires equal-sized inputs.
        img1, img2 = _ensure_same_size(img1, img2)

        # If block subdivision is requested, ensure dimensions are evenly
        # divisible by the block grid.
        block_shape = [block_height, block_width]
        if np.prod(block_shape) > 1:
            img1, img2 = _crop_to_block_divisible(
                img1, img2, block_height, block_width
            )
            print(
                f"Block subdivision: {block_height}x{block_width} grid, "
                f"working dimensions {img1.shape[1]}x{img1.shape[0]}"
            )

        # Total frames the recursion will produce, INCLUDING frame2, which the
        # generator excludes and we append. Everything downstream derives from
        # this rather than assuming 65, so changing times_to_interpolate moves
        # the selection with it instead of sampling the wrong positions.
        total_frames = 2 ** times_to_interpolate + 1

        # v2 unifies the return type: the old times_to_interpolate == 1 branch
        # returned a bare PNG, which cannot coexist with a BaseModel output.
        # That path now falls through like any other, yielding a 3-frame
        # sequence. The plugin always sends 6, so this only affects direct API
        # callers relying on upstream parity.
        selected = _select_indices(total_frames, num_views)
        selected_set = set(selected)

        print(
            f"Recursive interpolation: times_to_interpolate={times_to_interpolate}, "
            f"producing {total_frames} frames, returning {len(selected)} as PNG"
        )
        print(f"Selected source indices: {selected}")

        out_dir = PyPath(tempfile.mkdtemp())
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        preview_frames = []
        entries = []
        view_no = 0

        # STREAMING. Each frame is written to disk if selected, downscaled for
        # the preview, then dropped. Holding the full sequence in memory - as
        # v1 did - costs ~14.5 GB at 4000x4000 and cannot complete. See the
        # module docstring before changing this loop.
        for idx, frame in enumerate(
            chain(
                self._recursive_interpolate(
                    img1, img2, times_to_interpolate, block_shape
                ),
                (img2,),
            )
        ):
            preview_frames.append(_downscale_u8(frame, preview_long_edge))

            if idx in selected_set:
                # Filename carries BOTH the view ordinal and the true source
                # index, so ordering is self-describing and the consumer can
                # assert on it. Selected indices are not contiguous and are
                # unevenly spaced by necessity, so a bare sequence number would
                # lose the information needed to verify the selection.
                name = f"view_{view_no:02d}_src{idx:03d}.png"
                _write_image(str(frames_dir / name), frame)
                entries.append({"view": view_no, "source_index": idx, "file": name})
                view_no += 1

            del frame

        height, width = img1.shape[:2]
        manifest = {
            "schema": 1,
            "num_views": len(entries),
            "total_frames": total_frames,
            "times_to_interpolate": times_to_interpolate,
            "block_height": block_height,
            "block_width": block_width,
            "frame_width": int(width),
            "frame_height": int(height),
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

        preview_path = out_dir / "preview.mp4"
        mediapy.write_video(str(preview_path), preview_frames, fps=30)

        print(
            f"Wrote {len(entries)} PNG views ({width}x{height}) and a "
            f"{len(preview_frames)}-frame preview at "
            f"{preview_frames[0].shape[1]}x{preview_frames[0].shape[0]}"
        )

        return Output(preview=Path(preview_path), frames=Path(zip_path))
