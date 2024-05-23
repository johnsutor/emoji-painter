#!/usr/bin/env python3


from cog import BasePredictor, Input, Path
from omegaconf import OmegaConf

from inference import inference
from model import Network

CONFIG = OmegaConf.load("./configs/inference.yaml")


class Predictor(BasePredictor):
    def setup(self):
        self.model = Network(
            num_shapes=CONFIG.num_shapes,
            param_per_stroke=CONFIG.model.param_per_stroke,
            num_strokes=CONFIG.num_strokes,
            hidden_dim=CONFIG.model.hidden_dim,
            n_heads=CONFIG.model.n_heads,
            n_enc_layers=CONFIG.model.n_enc_layers,
            n_dec_layers=CONFIG.model.n_dec_layers,
        )

    def predict(
        self,
        target_path: Path = Input(description="Image to recreate"),
        scale: float = Input(description="Factor to scale image by", default=2),
    ) -> Path:
        """Run a single prediction on the model"""
        CONFIG.scale = scale
        CONFIG.target_path = target_path
        inference(CONFIG, self.model)
        return CONFIG.recreation_path
