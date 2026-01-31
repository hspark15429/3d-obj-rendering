"""
TRELLIS-VGGT pipeline wrapper for 3D reconstruction.

Wraps the TRELLIS-VGGT pipeline with our interface conventions.
Based on: https://github.com/estheryang11/ReconViaGen (app_refine.py)
Reference: https://huggingface.co/spaces/Stable-X/ReconViaGen/blob/main/app.py
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Any

import torch
from PIL import Image

# CRITICAL: Set SPCONV_ALGO before importing trellis
# This must be done before any trellis imports or spconv will hang/crash
os.environ['SPCONV_ALGO'] = 'native'

logger = logging.getLogger(__name__)

# Model paths
DEFAULT_MODEL_ID = "Stable-X/trellis-vggt-v0-2"
WEIGHTS_DIR = Path("/app/weights/reconviagen")


class TrellisPipelineWrapper:
    """
    Wrapper around TrellisVGGTTo3DPipeline for ReconViaGen.

    Based on estheryang11/ReconViaGen app_refine.py which uses
    optimized camera registration from VGGT.

    Handles:
    - Lazy model loading
    - Image preprocessing
    - Inference with configurable parameters
    - Mesh extraction and export
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda"):
        self._model_id = model_id
        self._device = device
        self._pipeline = None
        self._loaded = False

    def load(self) -> None:
        """Load the TRELLIS pipeline."""
        if self._loaded:
            return

        logger.info(f"Loading TRELLIS pipeline from {self._model_id}")

        try:
            # Import here to defer heavy imports
            from trellis.pipelines import TrellisVGGTTo3DPipeline

            # Load from HuggingFace or local cache
            local_path = WEIGHTS_DIR / "trellis-vggt"
            if local_path.exists():
                logger.info(f"Loading from local cache: {local_path}")
                self._pipeline = TrellisVGGTTo3DPipeline.from_pretrained(str(local_path))
            else:
                logger.info(f"Loading from HuggingFace: {self._model_id}")
                self._pipeline = TrellisVGGTTo3DPipeline.from_pretrained(self._model_id)

            self._pipeline.to(self._device)
            # CRITICAL: VGGT_model is not part of self.models, so Pipeline.to() doesn't move it
            # We must explicitly move it to the target device
            if hasattr(self._pipeline, 'VGGT_model') and self._pipeline.VGGT_model is not None:
                self._pipeline.VGGT_model.to(self._device)
                logger.info(f"Moved VGGT_model to {self._device}")
            self._loaded = True
            logger.info("TRELLIS pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load TRELLIS pipeline: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if pipeline is loaded."""
        return self._loaded

    def run(
        self,
        images: List[Image.Image],
        seed: int = 42,
        sparse_steps: int = 30,
        sparse_cfg: float = 7.5,
        slat_steps: int = 12,
        slat_cfg: float = 3.0,
    ) -> Tuple[dict, Any, Any]:
        """
        Run reconstruction on input images.

        Args:
            images: List of PIL Images (multi-view input)
            seed: Random seed for reproducibility
            sparse_steps: Diffusion steps for sparse structure
            sparse_cfg: CFG strength for sparse structure
            slat_steps: Diffusion steps for SLAT
            slat_cfg: CFG strength for SLAT

        Returns:
            Tuple of (outputs dict, video1, video2)
            outputs contains 'gaussian' and 'mesh' keys
        """
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        logger.info(f"Running TRELLIS inference on {len(images)} images")

        outputs, video1, video2 = self._pipeline.run(
            image=images,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=True,  # Let pipeline handle preprocessing
            sparse_structure_sampler_params={
                "steps": sparse_steps,
                "cfg_strength": sparse_cfg
            },
            slat_sampler_params={
                "steps": slat_steps,
                "cfg_strength": slat_cfg
            },
            mode="multidiffusion",
        )

        logger.info("TRELLIS inference complete")
        return outputs, video1, video2

    def cleanup(self) -> None:
        """Release GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._loaded = False
            torch.cuda.empty_cache()
            logger.info("TRELLIS pipeline unloaded")
