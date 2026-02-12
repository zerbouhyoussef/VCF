FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install venv dependencies, create venv, and upgrade pip in venv
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir virtualenv

RUN python3 -m virtualenv /root/envs/VCF

COPY requirements.txt .

RUN /root/envs/VCF/bin/pip install --no-cache-dir --upgrade pip \
    && /root/envs/VCF/bin/pip install --no-cache-dir jupyterlab \
    && /root/envs/VCF/bin/pip install -r requirements.txt \
    && /root/envs/VCF/bin/pip install --no-cache-dir ipykernel \
    && /root/envs/VCF/bin/python -m ipykernel install --user --name vcf --display-name "Python (VCF)"
    

COPY . .

EXPOSE 8888 80

CMD ["/root/envs/VCF/bin/python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
