FROM huggingface/transformers-pytorch-gpu:4.29.2
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /src
ENV PYTHONPATH="${PYTHONPATH}:${WORKDIR}"
ENV TRANSFORMERS_CACHE="/src/.cache/"

COPY requirements.txt $WORKDIR

RUN pip uninstall -y onnxruntime onnxruntime-gpu onnx

RUN apt-get update && apt upgrade -y && \
		apt-get install -y libsm6 libxrender1 libfontconfig1 libxext6 libgl1-mesa-glx ffmpeg && \
		pip install -U pip setuptools && \
		pip install -U --no-cache-dir -r requirements.txt

COPY . $WORKDIR

ENTRYPOINT [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1" ]
