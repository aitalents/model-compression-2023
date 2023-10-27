import os
from pathlib import Path

from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, OptimizationConfig


class OnnxModelOptimizer:
    def __init__(self, model, optimization_level: int = 99):
        self.model = model
        self.onnx_path: Path = Path("onnx")
        self.onnx_path.mkdir(parents=True, exist_ok=True)

        print('Выполняем оптимизацию графа')
        self.optimizer = ORTOptimizer.from_pretrained(model)
        self.optimization_config = OptimizationConfig(
            optimization_level=optimization_level
        )

        # dynamic quantization is currently only supported for CPUs
        device = os.getenv("DEVICE")
        if device == "cpu":
            print('Выполняем динамическую квантизацию для ЦПУ-инференса')
            self.dynamic_quantizer = ORTQuantizer.from_pretrained(model)
            self.dqconfig = AutoQuantizationConfig.avx512_vnni(
                is_static=False, per_channel=False
            )

    def graph_optimization(
        self, model_name: str, model: ORTModelForSequenceClassification
    ):
        # save weights
        optimized_model_path = self.onnx_path / model_name

        device = os.getenv("DEVICE")
        if device == "gpu":
            provider = "CUDAExecutionProvider"
        else:
            provider = "CPUExecutionProvider"

        file_name = "model_optimized.onnx"

        if not optimized_model_path.exists():
            self.model.save_pretrained(optimized_model_path)

            self.optimizer.optimize(
                save_dir=optimized_model_path,
                optimization_config=self.optimization_config,
            )

            # apply the quantization configuration to the model
            if device == "cpu":
                self.dynamic_quantizer.quantize(
                    save_dir=optimized_model_path,
                    quantization_config=self.dqconfig,
                )

                file_name = "model_quantized.onnx"

            # apply the quantization configuration to the model
            if device == "cpu":
                self.dynamic_quantizer.quantize(
                    save_dir=optimized_model_path,
                    quantization_config=self.dqconfig,
                )

                file_name = "model_quantized.onnx"

        model = ORTModelForSequenceClassification.from_pretrained(
            optimized_model_path,
            file_name=file_name,
            provider=provider,
        )
        return model

    def quantinization(self):
        """
        dynamic quantization is currently only supported for CPUs
        """
        ...
