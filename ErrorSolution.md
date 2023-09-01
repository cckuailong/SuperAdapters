## Error1

```
ValueError: The current `device_map` had weights offloaded to the disk. Please provide an 
`offload_folder` for them.
```

### Solution

Add offload_folder="offload"

```python
checkpoint = "facebook/opt-13b"
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, device_map="auto", offload_folder="offload", torch_dtype=torch.float16
)
```

### Reference

https://huggingface.co/blog/accelerate-large-models

## Warning1

```
The dtype of attention mask (torch.int64) is not bool
```

### Solution

This is a warning that can be ignored. The attention_mask of dtype int64 is passed by the generation method of transformers and not needed for our model.

### Reference

https://github.com/THUDM/ChatGLM-6B/issues/521
