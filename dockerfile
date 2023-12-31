FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN python -m pip install --upgrade pip
RUN pip install torchreid
RUN pip install gdown

WORKDIR /container_dir

RUN pip uninstall opencv-python
RUN pip install opencv-python==4.8.0.74

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 setup.py install && \
    cmake -B build . && cmake --build build --target install && ldconfig

RUN git clone https://github.com/KaiyangZhou/deep-person-reid.git && \
    cd deep-person-reid && \
    pip install -r requirements.txt 
