## How to Build Jetson Conainer 0.10.2, required for running on Jetson Thor New Hugging Face Models

Follow below the simplified and unofficial installation path to get vLLM 0.10.2 on Jetson using Dusty’s jetson-containers (which can build/pull an aarch64 image for JetPack).

git clone https://github.com/dusty-nv/jetson-containers

cd jetson-containers/

sudo apt install -y python3-venv
python3 -m venv ~/jc-venv
source ~/jc-venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt


Install the jetson-containers helper scripts (autotag, run, build) and set up defaults for your JetPack/CUDA stack. 
./install.sh

VLLM_VERSION=0.10.2 jetson-containers build vllm

This installation will take around 70 minutes, where all container images are created:
$ docker images
```
REPOSITORY                    TAG                                            IMAGE ID       CREATED             SIZE
vllm                          r38.2.aarch64-cu130-24.04-vllm                 9c713472a98f   3 minutes ago       38.4GB   # The last container created
vllm                          r38.2.aarch64-cu130-24.04-mistral_common       cdc5e90dbbec   9 minutes ago       33.7GB
vllm                          r38.2.aarch64-cu130-24.04-opencv               26927da57b46   11 minutes ago      33.4GB
vllm                          r38.2.aarch64-cu130-24.04-vulkan               8f0ea9334bb1   13 minutes ago      30.9GB
vllm                          r38.2.aarch64-cu130-24.04-opengl               d66dfe52d84b   15 minutes ago      29.5GB
vllm                          r38.2.aarch64-cu130-24.04-torch-memory-saver   1eaba6339526   15 minutes ago      29.5GB
vllm                          r38.2.aarch64-cu130-24.04-flashinfer           82f5b6bf14ee   18 minutes ago      29.5GB
vllm                          r38.2.aarch64-cu130-24.04-cudnn_frontend       b3eb2a06bdb0   19 minutes ago      28.7GB
vllm                          r38.2.aarch64-cu130-24.04-xgrammar             abe6fa688cd1   19 minutes ago      28.6GB
vllm                          r38.2.aarch64-cu130-24.04-mamba                810b3d64e3b5   20 minutes ago      28.4GB
vllm                          r38.2.aarch64-cu130-24.04-causalconv1d         c082cf6ccede   21 minutes ago      28GB
vllm                          r38.2.aarch64-cu130-24.04-ninja                f752edc1936e   22 minutes ago      27.9GB
vllm                          r38.2.aarch64-cu130-24.04-torchaudio           a39915c2d1d3   22 minutes ago      27.8GB
vllm                          r38.2.aarch64-cu130-24.04-torchcodec           e0f4d48794ca   22 minutes ago      27.8GB
vllm                          r38.2.aarch64-cu130-24.04-pyav                 508fa588e81f   22 minutes ago      27.8GB
vllm                          r38.2.aarch64-cu130-24.04-pybind11             7614f007d1a6   23 minutes ago      27.7GB
vllm                          r38.2.aarch64-cu130-24.04-ffmpeg_git           da9258048f6f   23 minutes ago      27.7GB
vllm                          r38.2.aarch64-cu130-24.04-llvm_20              19a9ffb761d4   23 minutes ago      27.6GB
vllm                          r38.2.aarch64-cu130-24.04-video-codec-sdk      f4cf4f52b6f6   24 minutes ago      26.2GB
vllm                          r38.2.aarch64-cu130-24.04-xformers             c31d9a40bc09   25 minutes ago      26.2GB
vllm                          r38.2.aarch64-cu130-24.04-flash-attention      553d1c051ae7   25 minutes ago      26GB
vllm                          r38.2.aarch64-cu130-24.04-cutlass              c7eb7ae7d623   26 minutes ago      24.2GB
vllm                          r38.2.aarch64-cu130-24.04-cuda-python          854963a46601   28 minutes ago      23.2GB
vllm                          r38.2.aarch64-cu130-24.04-diffusers            c8679c8bb580   29 minutes ago      23.1GB
vllm                          r38.2.aarch64-cu130-24.04-bitsandbytes         d1a69de1918f   29 minutes ago      23.1GB
vllm                          r38.2.aarch64-cu130-24.04-triton               770153cfacbc   30 minutes ago      22.9GB
vllm                          r38.2.aarch64-cu130-24.04-transformers         339571df4404   52 minutes ago      16.9GB
vllm                          r38.2.aarch64-cu130-24.04-rust                 01c059e85e0a   53 minutes ago      16.7GB
vllm                          r38.2.aarch64-cu130-24.04-huggingface_hub      7461ea79d469   54 minutes ago      15.6GB
vllm                          r38.2.aarch64-cu130-24.04-torchvision          dea925c8a5de   55 minutes ago      15.6GB
vllm                          r38.2.aarch64-cu130-24.04-pytorch_2.9          6bcacde8b898   55 minutes ago      15.5GB
vllm                          r38.2.aarch64-cu130-24.04-onnx                 3ef1ee3c363f   59 minutes ago      14GB
vllm                          r38.2.aarch64-cu130-24.04-cmake                4919ed7fe759   59 minutes ago      13.9GB
vllm                          r38.2.aarch64-cu130-24.04-numpy                bf5a23d861f9   59 minutes ago      13.8GB
vllm                          r38.2.aarch64-cu130-24.04-python               daa11b03c940   59 minutes ago      13.8GB
vllm                          r38.2.aarch64-cu130-24.04-nvshmem              3b34fa9aa03e   About an hour ago   13.6GB
vllm                          r38.2.aarch64-cu130-24.04-cudss                aa84266f5ee2   About an hour ago   12.3GB
vllm                          r38.2.aarch64-cu130-24.04-cusparselt           8817aed8e000   About an hour ago   12.1GB
vllm                          r38.2.aarch64-cu130-24.04-nvpl                 b2dd6f581399   About an hour ago   9.92GB
vllm                          r38.2.aarch64-cu130-24.04-gdrcopy              6e94b743a736   About an hour ago   9.51GB
vllm                          r38.2.aarch64-cu130-24.04-nccl                 cdaf439697bd   About an hour ago   9.51GB
vllm                          r38.2.aarch64-cu130-24.04-cudnn                8049d02d81d2   About an hour ago   8.2GB
vllm                          r38.2.aarch64-cu130-24.04-cuda_13.0            1716c1a875f9   About an hour ago   7.66GB
vllm                          r38.2.aarch64-cu130-24.04-build-essential      4ff1294c2ca3   About an hour ago   859MB
vllm                          r38.2.aarch64-cu130-24.04-pip_cache_cu130      c9df16f8134c   About an hour ago   859MB
ubuntu                        24.04                                          f4158f3f9981   11 days ago         101MB
```

create the container:
jetson-containers run -d --name vllm -p 8001:8000 $(autotag vllm)
```
Namespace(packages=['vllm'], prefer=['local', 'registry', 'build'], disable=[''], user='dustynv', output='/tmp/autotag', quiet=False, verbose=False)
-- L4T_VERSION=38.2.0  JETPACK_VERSION=7.0  CUDA_VERSION=13.0
-- Finding compatible container image for ['vllm']
vllm:r38.2.aarch64-cu130-24.04-vllm
V4L2_DEVICES: 
### ARM64 architecture detected
### Jetson Detected
SYSTEM_ARCH=tegra-aarch64
+ docker run --runtime nvidia --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics -it --rm --network host --shm-size=8g --volume /tmp/argus_socket:/tmp/argus_socket --volume /etc/enctune.conf:/etc/enctune.conf --volume /etc/nv_tegra_release:/etc/nv_tegra_release --volume /tmp/nv_jetson_model:/tmp/nv_jetson_model --volume /var/run/dbus:/var/run/dbus --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket --volume /var/run/docker.sock:/var/run/docker.sock --volume /home/rteixeira/Documents/jetson-containers/data:/data -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro --device /dev/snd -e PULSE_SERVER=unix:/run/user/1000/pulse/native -v /run/user/1000/pulse:/run/user/1000/pulse --device /dev/bus/usb --device /dev/i2c-0 --device /dev/i2c-1 --device /dev/i2c-2 --device /dev/i2c-3 --device /dev/i2c-4 --device /dev/i2c-5 --device /dev/i2c-6 --device /dev/i2c-7 --device /dev/i2c-8 --device /dev/i2c-9 -v /run/jtop.sock:/run/jtop.sock -d --name vllm -p 8001:8000 vllm:r38.2.aarch64-cu130-24.04-vllm
WARNING: Published ports are discarded when using host network mode
e5fc1794d0707c6bbce93fba19ac6cd4e2ed1233a0b71f9c6c380a7d71f18ce8
```

docker exec -it vllm bash (check version 0.10.2)
```
# vllm --version
/opt/venv/lib/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/opt/venv/lib/python3.12/site-packages/transformers/utils/hub.py:111: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 09-21 23:01:33 [__init__.py:241] Automatically detected platform cuda.
0.10.2+cu130
```



