#!/usr/bin/env python3


import tempfile

import torch
from cog import BasePredictor, Input, Path
from omegaconf import OmegaConf

from inference import inference
from model import Network

CONFIG = OmegaConf.load("./configs/inference.yaml")


class Predictor(BasePredictor):
    def setup(self):
        self.model = Network(
            num_shapes=CONFIG.num_shapes,
            param_per_stroke=CONFIG.param_per_stroke,
            num_strokes=CONFIG.num_strokes,
            hidden_dim=CONFIG.hidden_dim,
            n_heads=CONFIG.n_heads,
            n_enc_layers=CONFIG.n_enc_layers,
            n_dec_layers=CONFIG.n_dec_layers,
        )
        self.model.load_state_dict(torch.load(CONFIG.weights_path))
        self.shapes = torch.load(CONFIG.shapes_path)

    def predict(
        self,
        image: Path = Input(description="Image to recreate"),
        scale: float = Input(description="Factor to scale image by", default=2),
    ) -> Path:
        """Run a single prediction on the model"""
        CONFIG.scale = scale
        CONFIG.target_path = image
        CONFIG.recreation_path = Path(tempfile.mkdtemp()) / "output.png"
        print(OmegaConf.to_yaml(CONFIG))
        inference(CONFIG, self.model, self.shapes)
        return CONFIG.recreation_path
