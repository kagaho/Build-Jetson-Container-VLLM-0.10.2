# How to Build Jetson Container Latest **vLLM 0.10.2** on Jetson Thor

Follow below the *simplified and unofficial* installation path to get **vLLM 0.10.2** on Jetson using **Dusty’s jetson-containers** (which can build/pull an aarch64 image for JetPack).

```bash
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers/
```

```bash
sudo apt install -y python3-venv
python3 -m venv ~/jc-venv
source ~/jc-venv/bin/activate
```

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Install the **jetson-containers helper scripts** (`autotag`, `run`, `build`) and set up defaults for your JetPack/CUDA stack:  

```bash
./install.sh
```

```bash
VLLM_VERSION=0.10.2 jetson-containers build vllm
```

---

## Build Progress

This installation will take *around 70 minutes*, where all container images are created:

```bash
$ docker images
```

```
REPOSITORY                    TAG                                            IMAGE ID       CREATED             SIZE
vllm                          r38.2.aarch64-cu130-24.04-vllm                 9c713472a98f   3 minutes ago       38.4GB   # The last container created
...
ubuntu                        24.04                                          f4158f3f9981   11 days ago         101MB
```

---

## Create the Container

```bash
$ jetson-containers run -d --name vllm -p 8001:8000 $(autotag vllm)
```

or  

```bash
$ docker run -d --name vllm_0_10_2 -it --gpus all -p 8001:8000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  vllm:r38.2.aarch64-cu130-24.04-vllm
```

Check version inside the container:  

```bash
docker exec -it vllm bash
# vllm --version
```

```
/opt/venv/lib/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated...
INFO 09-21 23:01:33 [__init__.py:241] Automatically detected platform cuda.
0.10.2+cu130
```

---

## Test Any Model (inside the container)

```bash
/# huggingface-cli login
```

```
⚠️  Warning: 'huggingface-cli login' is deprecated. Use 'hf auth login' instead.
...
Login successful.
The current active token is: `blabla`
```

---

Run the model:  

```bash
/# vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --enforce-eager --gpu-memory-utilization 0.85
```

(Logs will appear with model loading details, tensor setup, etc.)  

---

## Testing

```bash
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "meta-llama/Llama-3.1-8B-Instruct",
      "prompt": "Write a short story about a robot living on Mars.",
      "max_tokens": 100
  }'
```

**Example output:**

```json
{"id":"cmpl-28ddec6d2f5d4a42954f820c6dbdc0a3","object":"text_completion",...}
```

---

## At the Container Logs

```bash
(EngineCore_0 pid=454) WARNING 09-21 21:43:43 [cudagraph_dispatcher.py:102] cudagraph dispatching keys are not initialized...
(APIServer pid=387) INFO:     172.17.0.1:57744 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=387) INFO 09-21 21:43:53 [loggers.py:123] Engine 000: Avg prompt throughput: 1.2 tokens/s...
```
