## 报错1

```
ValueError: The current `device_map` had weights offloaded to the disk. Please provide an 
`offload_folder` for them.
```

### 解决方案

添加 offload_folder="offload"

```python
checkpoint = "facebook/opt-13b"
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, device_map="auto", offload_folder="offload", torch_dtype=torch.float16
)
```

### 参考链接

https://huggingface.co/blog/accelerate-large-models

