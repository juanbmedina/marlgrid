FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# System dependencies
RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Additional dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git


# RUN pip install protobuf==3.20.*
# Downgrade pip to avoid gym installation issues
RUN pip install --upgrade pip==23.3

# Install core Python packages
RUN pip install protobuf==3.20.3 gym==0.20.0

# Install MARLlib (do it after gym so it doesn't upgrade gym again)
RUN pip install marllib

WORKDIR /workspace

# Default command
CMD ["bash"]


