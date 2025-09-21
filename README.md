## How to Build Jetson Conainer latest VLLM 0.10.2 on Jetson Thor 

Follow below the simplified and unofficial installation path to get vLLM 0.10.2 on Jetson using Dusty’s jetson-containers (which can build/pull an aarch64 image for JetPack).

```
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers/
```
```
sudo apt install -y python3-venv
python3 -m venv ~/jc-venv
source ~/jc-venv/bin/activate
```
```
pip install --upgrade pip
pip install -r requirements.txt
```

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

Create the container:

$ jetson-containers run -d --name vllm -p 8001:8000 $(autotag vllm)

or 

$ docker run -d --name vllm_0_10_2 -it --gpus all -p 8001:8000 -v $HOME/.cache/huggingface:/root/.cache/huggingface vllm:r38.2.aarch64-cu130-24.04-vllm


docker 
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

#### Test any model (following commands inside the containe):


root@8cc52f8c9b13:/# huggingface-cli login
```
⚠️  Warning: 'huggingface-cli login' is deprecated. Use 'hf auth login' instead.

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible): 
Add token as git credential? (Y/n) n
Token is valid (permission: write).
The token `blabla` has been saved to /data/models/huggingface/stored_tokens
Your token has been saved to /data/models/huggingface/token
Login successful.
The current active token is: `blabla`
```


root@8cc52f8c9b13:/# vllm serve meta-llama/Llama-3.1-8B-Instruct   --host 0.0.0.0 --port 8000   --enforce-eager --gpu-memory-utilization 0.85
```
/opt/venv/lib/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/opt/venv/lib/python3.12/site-packages/transformers/utils/hub.py:111: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 09-21 21:35:57 [__init__.py:241] Automatically detected platform cuda.
(APIServer pid=387) INFO 09-21 21:35:59 [api_server.py:1882] vLLM API server version 0.10.2
(APIServer pid=387) INFO 09-21 21:35:59 [utils.py:328] non-default args: {'model_tag': 'meta-llama/Llama-3.1-8B-Instruct', 'host': '0.0.0.0', 'model': 'meta-llama/Llama-3.1-8B-Instruct', 'enforce_eager': True, 'gpu_memory_utilization': 0.85}
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 855/855 [00:00<00:00, 6.19MB/s]
(APIServer pid=387) INFO 09-21 21:36:07 [__init__.py:745] Resolved architecture: LlamaForCausalLM
(APIServer pid=387) `torch_dtype` is deprecated! Use `dtype` instead!
(APIServer pid=387) INFO 09-21 21:36:07 [__init__.py:1778] Using max model len 131072
(APIServer pid=387) INFO 09-21 21:36:08 [scheduler.py:222] Chunked prefill is enabled with max_num_batched_tokens=2048.
(APIServer pid=387) INFO 09-21 21:36:08 [__init__.py:3643] Cudagraph is disabled under eager mode
tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 55.4k/55.4k [00:00<00:00, 27.1MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 9.09M/9.09M [00:00<00:00, 16.9MB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 296/296 [00:00<00:00, 2.11MB/s]
generation_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 184/184 [00:00<00:00, 2.37MB/s]
/opt/venv/lib/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/opt/venv/lib/python3.12/site-packages/transformers/utils/hub.py:111: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 09-21 21:36:14 [__init__.py:241] Automatically detected platform cuda.
(EngineCore_0 pid=454) INFO 09-21 21:36:16 [core.py:648] Waiting for init message from front-end.
(EngineCore_0 pid=454) INFO 09-21 21:36:16 [core.py:75] Initializing a V1 LLM engine (v0.10.2) with config: model='meta-llama/Llama-3.1-8B-Instruct', speculative_config=None, tokenizer='meta-llama/Llama-3.1-8B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=131072, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, kv_cache_dtype=auto, device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=meta-llama/Llama-3.1-8B-Instruct, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, pooler_config=None, compilation_config={"level":0,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":null,"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"cudagraph_mode":0,"use_cudagraph":true,"cudagraph_num_of_warmups":0,"cudagraph_capture_sizes":[],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"pass_config":{},"max_capture_size":0,"local_cache_dir":null}
[W921 21:36:17.897675918 ProcessGroupNCCL.cpp:941] Warning: TORCH_NCCL_AVOID_RECORD_STREAMS is the default now, this environment variable is thus deprecated. (function operator())
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(EngineCore_0 pid=454) INFO 09-21 21:36:17 [parallel_state.py:1134] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
(EngineCore_0 pid=454) INFO 09-21 21:36:17 [topk_topp_sampler.py:58] Using FlashInfer for top-p & top-k sampling.
(EngineCore_0 pid=454) INFO 09-21 21:36:17 [gpu_model_runner.py:1921] Starting to load model meta-llama/Llama-3.1-8B-Instruct...
(EngineCore_0 pid=454) INFO 09-21 21:36:17 [gpu_model_runner.py:1953] Loading model from scratch...
(EngineCore_0 pid=454) INFO 09-21 21:36:17 [cuda.py:328] Using Flash Attention backend on V1 engine.
(EngineCore_0 pid=454) INFO 09-21 21:36:18 [weight_utils.py:304] Using model weights format ['*.safetensors']
model-00004-of-00004.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████| 1.17G/1.17G [01:19<00:00, 14.8MB/s]
model-00003-of-00004.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████| 4.92G/4.92G [01:59<00:00, 41.3MB/s]
model-00001-of-00004.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████| 4.98G/4.98G [01:59<00:00, 41.6MB/s]
model-00002-of-00004.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████| 5.00G/5.00G [01:59<00:00, 41.8MB/s]
(EngineCore_0 pid=454) INFO 09-21 21:38:18 [weight_utils.py:325] Time spent downloading weights for meta-llama/Llama-3.1-8B-Instruct: 120.139347 secondsMB/s]
model.safetensors.index.json: 100%|█████████████████████████████████████████████████████████████████████████████████████| 23.9k/23.9k [00:00<00:00, 57.3MB/s]
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:02<00:08,  2.82s/it]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:03<00:02,  1.34s/it]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:03<00:01,  1.10s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:04<00:00,  1.05s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:04<00:00,  1.22s/it]
(EngineCore_0 pid=454) 
(EngineCore_0 pid=454) INFO 09-21 21:38:24 [default_loader.py:267] Loading weights took 5.04 seconds
(EngineCore_0 pid=454) INFO 09-21 21:38:24 [gpu_model_runner.py:1975] Model loading took 14.9889 GiB and 126.264947 seconds
(EngineCore_0 pid=454) INFO 09-21 21:38:25 [gpu_worker.py:276] Available KV cache memory: 85.79 GiB
(EngineCore_0 pid=454) INFO 09-21 21:38:26 [kv_cache_utils.py:850] GPU KV cache size: 702,800 tokens
(EngineCore_0 pid=454) INFO 09-21 21:38:26 [kv_cache_utils.py:854] Maximum concurrency for 131,072 tokens per request: 5.36x
(EngineCore_0 pid=454) INFO 09-21 21:38:31 [core.py:217] init engine (profile, create kv cache, warmup model) took 6.84 seconds
(EngineCore_0 pid=454) INFO 09-21 21:38:32 [__init__.py:3643] Cudagraph is disabled under eager mode
(APIServer pid=387) INFO 09-21 21:38:32 [loggers.py:142] Engine 000: vllm cache_config_info with initialization after num_gpu_blocks is: 43925
(APIServer pid=387) INFO 09-21 21:38:32 [async_llm.py:166] Torch profiler disabled. AsyncLLM CPU traces will not be collected.
(APIServer pid=387) INFO 09-21 21:38:32 [api_server.py:1680] Supported_tasks: ['generate']
(APIServer pid=387) WARNING 09-21 21:38:32 [__init__.py:1678] Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=387) INFO 09-21 21:38:32 [serving_responses.py:126] Using default chat sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=387) INFO 09-21 21:38:32 [serving_chat.py:137] Using default chat sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=387) INFO 09-21 21:38:32 [serving_completion.py:79] Using default completion sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=387) INFO 09-21 21:38:32 [api_server.py:1957] Starting vLLM API server 0 on http://0.0.0.0:8000
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:36] Available routes are:
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /openapi.json, Methods: HEAD, GET
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /docs, Methods: HEAD, GET
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /docs/oauth2-redirect, Methods: HEAD, GET
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /redoc, Methods: HEAD, GET
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /health, Methods: GET
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /load, Methods: GET
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /ping, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /ping, Methods: GET
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /tokenize, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /detokenize, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /v1/models, Methods: GET
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /version, Methods: GET
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /v1/responses, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /v1/chat/completions, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /v1/completions, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /v1/embeddings, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /pooling, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /classify, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /score, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /v1/score, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /v1/audio/transcriptions, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /v1/audio/translations, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /rerank, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /v1/rerank, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /v2/rerank, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /invocations, Methods: POST
(APIServer pid=387) INFO 09-21 21:38:32 [launcher.py:44] Route: /metrics, Methods: GET
(APIServer pid=387) INFO:     Started server process [387]
(APIServer pid=387) INFO:     Waiting for application startup.
(APIServer pid=387) INFO:     Application startup complete.
```


Testing:

curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Write a short story about a robot living on Mars.",
    "max_tokens": 100
  }'
```
{"id":"cmpl-28ddec6d2f5d4a42954f820c6dbdc0a3","object":"text_completion","created":1758491021,"model":"meta-llama/Llama-3.1-8B-Instruct","choices":[{"index":0,"text":" \nThe sun was setting on Mars, casting a reddish hue over the barren landscape. Zeta, a small robot, sat on the edge of a rocky outcropping, watching the sky turn pink and orange. It was a beautiful sight, one that Zeta never got tired of seeing.\nZeta had been living on Mars for several years now, sent by its creators to explore the planet and gather data. It had been a long and lonely journey, but Zeta had grown to love","logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null,"prompt_logprobs":null,"prompt_token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":12,"total_tokens":112,"completion_tokens":100,"prompt_tokens_details":null},"kv_transfer_params":null}
```

At the container:
```
(EngineCore_0 pid=454) WARNING 09-21 21:43:43 [cudagraph_dispatcher.py:102] cudagraph dispatching keys are not initialized. No cudagraph will be used.
(APIServer pid=387) INFO:     172.17.0.1:57744 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=387) INFO 09-21 21:43:53 [loggers.py:123] Engine 000: Avg prompt throughput: 1.2 tokens/s, Avg generation throughput: 10.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=387) INFO 09-21 21:44:03 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```
