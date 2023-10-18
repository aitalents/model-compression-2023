FROM huggingface/transformers-pytorch-gpu:4.27.0

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /src
ENV PYTHONPATH="${PYTHONPATH}:${WORKDIR}"

COPY requirements.txt $WORKDIR

RUN apt-get update && apt upgrade -y && \
		apt-get install -y libsm6 libxrender1 libfontconfig1 libxext6 libgl1-mesa-glx ffmpeg && \
		pip install -U pip setuptools && \
		pip install -U --no-cache-dir -r requirements.txt

COPY . $WORKDIR

CMD jupyter notebook --ip 0.0.0.0 --port 9988 --allow-root --NotebookApp.token=""
