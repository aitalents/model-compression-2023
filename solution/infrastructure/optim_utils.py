from pathlib import Path

from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, OptimizationConfig


class OnnxModelOptimizer:

    def __init__(
            self,
            model,
            optimization_level: int = 99
            ):
        self.model = model
        self.onnx_path: Path = Path("onnx")
        self.onnx_path.mkdir(parents=True, exist_ok=True)

        self.optimizer = ORTOptimizer.from_pretrained(model)
        self.optimization_config = OptimizationConfig(optimization_level=optimization_level)

        self.dynamic_quantizer = ORTQuantizer.from_pretrained(model)
        self.dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    def graph_optimization(self, model_name: str):
        # save weights
        optimized_model_path = self.onnx_path/model_name
        if not optimized_model_path.exists():
            self.model.save_pretrained(optimized_model_path)

            self.optimizer.optimize(
                    save_dir=optimized_model_path,
                    optimization_config=self.optimization_config,
            )
            # apply the quantization configuration to the model
            self.dynamic_quantizer.quantize(
                 save_dir=optimized_model_path,
                 quantization_config=self.dqconfig,
            )

        model = ORTModelForSequenceClassification.from_pretrained(
                optimized_model_path,
                file_name="model_quantized.onnx",
                provider="CPUExecutionProvider"
        )
        return model
