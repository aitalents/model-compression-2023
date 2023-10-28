from models import TransformerTextClassificationModel
from optimum.bettertransformer import BetterTransformer
import torch.nn.utils.prune as prune
import sys
import torch
import torch.nn as nn
import time
sys.path.append('../')
from configs.config import AppConfig, ModelConfig

config = AppConfig.parse_file("../configs/app_config.yaml")
models = [
            TransformerTextClassificationModel(conf.model, conf.model_path, conf.tokenizer)
            for conf in config.models
        ]
for model in models:
    inputs = model.tokenize_texts(['i love this film'])
    start_time = time.time()
    for i in range(100):
        outputs = model(inputs)
    print(time.time() - start_time)
