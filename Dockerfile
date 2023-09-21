FROM huggingface/transformers-pytorch-gpu

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /src
ENV PYTHONPATH="${PYTHONPATH}:${WORKDIR}"

COPY requirements.txt $WORKDIR

RUN pip install -U pip setuptools && \
    pip install -U --no-cache-dir -r requirements.txt

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=''", "--allow-root"]
