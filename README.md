# SuperAdapters

Finetune ALL LLMs with ALL Adapeters on ALL Platforms!

## Support

| Model        | LoRA | Prefix Tuning  | P-Tuning | Prompt Tuning  |
|--------------| ---- | ---- | ---- | ---- |
| Bloom        | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| LLaMA        | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| ChatGLM      | :white_check_mark: | :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: |
| ChatGLM2     | :white_check_mark: | :ballot_box_with_check: |️ :ballot_box_with_check: | :ballot_box_with_check: |

You can Finetune LLM on 
- Windows
- Linux
- Mac M1/2

## Requirement

CentOS:

```bash
yum install -y xz-devel
```

Ubuntu:
```bash
apt-get install -y liblzma-dev
```

MacOS:
```bash
brew install xz
```

If you want to use gpu on Mac, Please read [How to use GPU on Mac](./MacGPUEnv.md)

```shell
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install -r requirements.txt
```

## LLMs

| Model        | Download Link |
|--------------| ---- |
| Bloom        | [https://huggingface.co/bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m) |
| LLaMA        | [https://huggingface.co/openlm-research/open_llama_3b_600bt_preview](https://huggingface.co/openlm-research/open_llama_3b_600bt_preview) |
| Vicuna       | [https://huggingface.co/lmsys/vicuna-7b-delta-v1.1](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1) |
| ChatGLM      | [https://huggingface.co/THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b) |
| ChatGLM2      | [https://huggingface.co/THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b) |

## Usage

### ChatGLM with lora

```bash
python finetune.py --model_type chatglm --data "data/train/" --model_path "LLMs/chatglm/chatglm-6b/" --adapter "lora" --output_dir "output/chatglm"
```

```bash
python generate.py --model_type chatglm --instruction "Who are you?" --model_path "LLMs/chatglm/chatglm-6b/" --adapter_weights "output/chatglm" --max_new_tokens 256
```

### LLaMa with lora

```bash
python finetune.py --model_type llama --data "data/train/" --model_path "LLMs/open-llama/open-llama-3b/" --adapter "lora" --output_dir "output/llama"
```

```bash
python generate.py --model_type llama --instruction "Who are you?" --model_path "LLMs/open-llama/open-llama-3b" --adapter_weights "output/llama" --max_new_tokens 256
```

### Bloom with lora

```bash
python finetune.py --model_type bloom --data "data/train/" --model_path "LLMs/bloom/bloomz-560m" --adapter "lora" --output_dir "output/bloom"
```

```bash
python generate.py --model_type bloom --instruction "Who are you?" --model_path "LLMs/bloom/bloomz-560m" --adapter_weights "output/bloom" --max_new_tokens 256
```

## Params

### Finetune

```shell
usage: finetune.py [-h] [--data [DATA [DATA ...]]] [--model_type {llama,chatglm,bloom,moss}] [--model_path MODEL_PATH] [--output_dir OUTPUT_DIR] [--adapter {lora,adalora,prompt,p_tuning,prefix}]
                   [--lora_r LORA_R] [--lora_alpha LORA_ALPHA] [--lora_dropout LORA_DROPOUT] [--lora_target_modules LORA_TARGET_MODULES [LORA_TARGET_MODULES ...]] [--adalora_init_r ADALORA_INIT_R]
                   [--adalora_tinit ADALORA_TINIT] [--adalora_tfinal ADALORA_TFINAL] [--adalora_delta_t ADALORA_DELTA_T] [--num_virtual_tokens NUM_VIRTUAL_TOKENS] [--mapping_hidden_dim MAPPING_HIDDEN_DIM]
                   [--epochs EPOCHS] [--learning_rate LEARNING_RATE] [--cutoff_len CUTOFF_LEN] [--val_set_size VAL_SET_SIZE] [--group_by_length] [--logging_steps LOGGING_STEPS] [--load_8bit]
                   [--add_eos_token] [--resume_from_checkpoint [RESUME_FROM_CHECKPOINT]] [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE] [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]

Process some integers.

optional arguments:
  -h, --help            show this help message and exit
  --data [DATA [DATA ...]]
                        the data used for instructing tuning
  --model_type {llama,chatglm,bloom,moss}
  --model_path MODEL_PATH
  --output_dir OUTPUT_DIR
                        The DIR to save the model
  --disable_wandb Disable report to wandb
  --adapter {lora,adalora,prompt,p_tuning,prefix}
  --lora_r LORA_R
  --lora_alpha LORA_ALPHA
  --lora_dropout LORA_DROPOUT
  --lora_target_modules LORA_TARGET_MODULES [LORA_TARGET_MODULES ...]
                        the module to be injected, e.g. q_proj/v_proj/k_proj/o_proj for llama, query_key_value for bloom&GLM
  --adalora_init_r ADALORA_INIT_R
  --adalora_tinit ADALORA_TINIT
                        number of warmup steps for AdaLoRA wherein no pruning is performed
  --adalora_tfinal ADALORA_TFINAL
                        fix the resulting budget distribution and fine-tune the model for tfinal steps when using AdaLoRA
  --adalora_delta_t ADALORA_DELTA_T
                        interval of steps for AdaLoRA to update rank
  --num_virtual_tokens NUM_VIRTUAL_TOKENS
  --mapping_hidden_dim MAPPING_HIDDEN_DIM
  --epochs EPOCHS
  --learning_rate LEARNING_RATE
  --cutoff_len CUTOFF_LEN
  --val_set_size VAL_SET_SIZE
  --group_by_length
  --logging_steps LOGGING_STEPS
  --load_8bit
  --add_eos_token
  --resume_from_checkpoint [RESUME_FROM_CHECKPOINT]
                        resume from the specified or the latest checkpoint, e.g. `--resume_from_checkpoint [path]` or `--resume_from_checkpoint`
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
```

## Generate

```shell
usage: generate.py [-h] [--instruction INSTRUCTION] [--input INPUT] [--model_type {llama,chatglm,bloom,moss}] [--model_path MODEL_PATH] [--adapter_weights ADAPTER_WEIGHTS] [--load_8bit]
                   [--temperature TEMPERATURE] [--top_p TOP_P] [--top_k TOP_K] [--max_new_tokens MAX_NEW_TOKENS]

Process some integers.

optional arguments:
  -h, --help            show this help message and exit
  --instruction INSTRUCTION
  --input INPUT
  --data The DIR of test data
  --model_type {llama,chatglm,bloom,moss}
  --model_path MODEL_PATH
  --adapter_weights ADAPTER_WEIGHTS
                        The DIR of adapter weights
  --load_8bit
  --temperature TEMPERATURE
                        temperature higher, LLM is more creative
  --top_p TOP_P
  --top_k TOP_K
  --max_new_tokens MAX_NEW_TOKENS
```

## Reference

- https://github.com/AGI-Edgerunners/LLM-Adapters
- https://github.com/PhoebusSi/Alpaca-CoT
