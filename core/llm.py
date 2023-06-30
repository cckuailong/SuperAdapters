import os
import sys
import torch
from peft import (
    AdaLoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    LoraConfig,
    TaskType,
    get_peft_model,
)

from typing import List


class LLM:
    # base params
    base_model: str = "LLMs/chatglm/chatglm-6b"
    model_type: str = "chatglm"
    data_path: str = "data/train"
    output_dir: str = "./output"
    disable_wandb: bool = False

    # adapter params
    adapter: str = "prefix"
    adapter_weights: str = "output/chatglm"

    # lora hyperparams
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = [
        "query_key_value"
    ]

    # adalora hyperparams
    adalora_init_r: int = 12
    adalora_tinit: int = 200
    adalora_tfinal: int = 1000
    adalora_delta_t: int = 10

    # prefix/prompt tuning/ptuning hyperparams
    num_virtual_tokens: int = 32
    mapping_hidden_dim: int = 1024

    # training hyperparams
    epochs: int = 3
    learning_rate: float = 3e-4
    cutoff_len: int = 256
    val_set_size: float = 0.15
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    logging_steps: int = 10
    load_8bit: bool = False
    add_eos_token: bool = False
    resume_from_checkpoint: str = None  # either training checkpoint or final adapter
    per_gpu_train_batch_size: int = 4
    gradient_accumulation_steps: int = 32

    # auto set, user cannot control
    device: str = None
    use_mps_device: bool = False
    is_fp16: bool = True
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    # generate
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_new_tokens: int = 512

    def load_adapter_config(self, model):
        if self.adapter == "lora":
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                bias="none",
                inference_mode=False,
            )
        elif self.adapter == 'adalora':
            config = AdaLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                init_r=self.adalora_init_r,
                r=self.lora_r,
                beta1=0.85,
                beta2=0.85,
                tinit=self.adalora_tinit,
                tfinal=self.adalora_tfinal,
                deltaT=self.adalora_delta_t,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                inference_mode=False,
            )
        elif self.adapter == "prefix":
            config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.num_virtual_tokens,
                encoder_hidden_size=self.mapping_hidden_dim,
                prefix_projection=True
            )
        elif self.adapter == "p_tuning":
            config = PromptEncoderConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.num_virtual_tokens,
                encoder_hidden_size=self.mapping_hidden_dim
            )
        elif self.adapter == "prompt":
            config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.num_virtual_tokens
            )
        else:
            raise KeyError("Unknow adapter: {}".format(self.adapter))

        model = get_peft_model(model, config)
        model.print_trainable_parameters()

        return model

    def auto_device(self):
        if not self.device:
            try:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif sys.platform == "darwin" and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except:
                self.device = "cpu"

        if self.device == "mps":
            self.use_mps_device = True
            self.is_fp16 = False
            self.device_map = {"": self.device}
        else:
            if self.load_8bit:
                self.is_fp16 = False
            self.device_map = "auto"
