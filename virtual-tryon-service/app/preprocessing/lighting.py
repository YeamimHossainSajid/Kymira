"""
virtual-tryon-service/app/preprocessing/lighting.py

Exposure and luminance normalization filter for harmonizing garment captures
with outdoor/ambient model lighting conditions.
"""

from typing import Optional
import cv2
import numpy as np


class LightingNormalizer:
    """
    Handles brightness, luminance calibration, and exposure curve alignment
    between human model images and studio garment cutouts.
    """

    def __init__(self, target_clip_min: float = 0.0, target_clip_max: float = 255.0):
        self.clip_min = target_clip_min
        self.clip_max = target_clip_max

    def compute_average_luminance(
        self,
        image_bgr: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Computes the mean lightness (L channel in LAB color space).
        If a mask is supplied, the calculation is strictly constrained to the masked area.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Invalid or empty image provided.")

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        if mask is not None:
            # Ensure binary single-channel mask
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mean_val = cv2.mean(l_channel, mask=mask)[0]
        else:
            mean_val = float(np.mean(l_channel))

        return mean_val

    def normalize_garment_exposure(
        self,
        model_bgr: np.ndarray,
        garment_bgr: np.ndarray,
        model_mask: Optional[np.ndarray] = None,
        garment_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Adjusts the garment's exposure curve to match the reference model's ambient lighting.
        Preserves color fidelity (A & B channels) while adjusting the L (Lightness) channel.
        """
        model_luminance = self.compute_average_luminance(model_bgr, model_mask)
        garment_luminance = self.compute_average_luminance(garment_bgr, garment_mask)

        # Avoid zero division on fully black garment inputs
        if garment_luminance <= 1e-5:
            return garment_bgr.copy()

        # Compute dynamic gain ratio
        gain_ratio = model_luminance / garment_luminance

        # Convert garment to LAB
        garment_lab = cv2.cvtColor(garment_bgr, cv2.COLOR_BGR2LAB)
        garment_l = garment_lab[:, :, 0].astype(np.float32)

        # Apply gain curve and clamp boundaries
        adjusted_l = np.clip(garment_l * gain_ratio, self.clip_min, self.clip_max).astype(np.uint8)

        # Reconstruct normalized image
        garment_lab[:, :, 0] = adjusted_l
        normalized_bgr = cv2.cvtColor(garment_lab, cv2.COLOR_LAB2BGR)

        return normalized_bgr


# Module-level convenience function
def normalize_lighting(
    model_bgr: np.ndarray,
    garment_bgr: np.ndarray,
    model_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    normalizer = LightingNormalizer()
    return normalizer.normalize_garment_exposure(model_bgr, garment_bgr, model_mask=model_mask)