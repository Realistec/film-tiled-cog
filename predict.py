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
"""

import os
import tempfile
from pathlib import Path as PyPath
from typing import List, Optional

import numpy as np
import tensorflow as tf
from PIL import Image, PngImagePlugin
import mediapy
from cog import BasePredictor, Input, Path

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
    ) -> Path:
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

        if times_to_interpolate == 1:
            # Single mid-frame: return as PNG.
            time = np.array([0.5], dtype=np.float32)
            mid = self._interpolate_with_blocks(
                img1[np.newaxis, ...],
                img2[np.newaxis, ...],
                time,
                block_shape,
            )[0]

            out_dir = PyPath(tempfile.mkdtemp())
            out_path = out_dir / "out.png"
            _write_image(str(out_path), mid)
            return Path(out_path)

        # Multi-frame: recursively interpolate, then encode as MP4.
        print(
            f"Recursive interpolation: times_to_interpolate={times_to_interpolate}, "
            f"will produce {2 ** times_to_interpolate + 1} frames"
        )
        frames = list(
            self._recursive_interpolate(img1, img2, times_to_interpolate, block_shape)
        )
        # The recursive generator excludes frame2; append it for the final video.
        frames.append(img2)
        # Convert from [0,1] float32 to uint8 for mediapy.
        frames_u8 = [
            np.clip(f * _UINT8_MAX_F + 0.5, 0, 255).astype(np.uint8) for f in frames
        ]

        out_dir = PyPath(tempfile.mkdtemp())
        out_path = out_dir / "out.mp4"
        mediapy.write_video(str(out_path), frames_u8, fps=30)
        return Path(out_path)
