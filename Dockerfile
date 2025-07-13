FROM tensorflow/tensorflow:latest

# --- Dépendances système ---
RUN apt-get update -y && apt-get install -y \
    git \
    ffmpeg \
    build-essential cmake pkg-config \
    libjpeg-dev libtiff-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev gfortran \
    python3-dev python3-pip python3-tk \
    wget unzip vim \
    && apt-get clean

# --- OpenCV 4.5.5 via pip ---
RUN pip install --no-cache-dir \
    opencv-python==4.5.5.64 \
    opencv-contrib-python==4.5.5.64

# --- dlib  compilation ---
RUN pip install numpy
RUN git clone https://github.com/davisking/dlib.git /opt/dlib
WORKDIR /opt/dlib
RUN python setup.py install


# --- Dépendances Python ---
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# --- Fichiers de travail ---
COPY . /app
WORKDIR /app


# CMD par défaut
CMD ["python", "main.py"]
