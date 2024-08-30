# Use vllm to accelerate inference

## Start vllm server

1. Open the "nvidia-runtime"

vim /etc/docker/daemon.json

Add the following content
```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

Restart docker
```
systemctl daemon-reload
systemctl restart docker
```

2. Install nvidia-container-runtime and nvidia-docker2

Run
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.repo | \
sudo tee /etc/yum.repos.d/nvidia-container-runtime.repo
```

Run
```
yum install nvidia-container-runtime nvidia-docker2 -y
```

Restart docker
```
systemctl restart docker
```

3. Start vllm Server

```
docker run -d --runtime nvidia --gpus all -v /root/SuperAdapters/output/llama3.1-combined:/root/SuperAdapters/output/llama3.1-combined -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model /root/SuperAdapters/output/llama3.1-combined --trust-remote-code
```
