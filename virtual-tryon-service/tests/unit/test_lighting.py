import sys
from pathlib import Path

# Pridá cestu k 'virtual-tryon-service' do sys.path pred importom app
SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SERVICE_ROOT))

import numpy as np
from app.preprocessing.lighting import LightingNormalizer


def test_exposure_normalization():
    normalizer = LightingNormalizer()

    # High-luminance model (outdoor lighting: value ~210)
    model = np.full((100, 100, 3), 210, dtype=np.uint8)

    # Garment photographed in a darker studio (value ~70)
    garment = np.full((100, 100, 3), 70, dtype=np.uint8)

    # Run the normalization
    harmonized = normalizer.normalize_garment_exposure(model, garment)

    orig_l = normalizer.compute_average_luminance(garment)
    new_l = normalizer.compute_average_luminance(harmonized)
    model_l = normalizer.compute_average_luminance(model)

    print(f"Original garment luminance: {orig_l:.2f}")
    print(f"Target model luminance: {model_l:.2f}")
    print(f"Resulting garment luminance: {new_l:.2f}")

    assert new_l > orig_l, "Garment luminance should have increased"
    assert abs(new_l - model_l) < 5.0, "Resulting luminance should match the reference model"


if __name__ == "__main__":
    test_exposure_normalization()
    print("Test passed successfully!")