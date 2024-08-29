import os
import sys
import re
import copy
import torch
import transformers

from common.base import IGNORE_INDEX
from common.prompt import PROMPT_DICT

from transformers import (
    BloomTokenizerFast,
    BloomForSequenceClassification,
    BitsAndBytesConfig
)
from safetensors.torch import load_file

from peft import (
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel
)

from tqdm import tqdm
from core.llm import LLM


class BLoomClassify(LLM):
    def __init__(self):
        if not self.lora_target_modules:
            self.lora_target_modules = [
                "query_key_value"
            ]

    def get_model_tokenizer(self):
        if self.adapter == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model = BloomForSequenceClassification.from_pretrained(
                self.base_model,
                load_in_8bit=self.load_8bit,
                device_map=self.device_map,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
            )
        else:
            model = BloomForSequenceClassification.from_pretrained(
                self.base_model,
                load_in_8bit=self.load_8bit,
                device_map=self.device_map,
                low_cpu_mem_usage=True,
            )
        tokenizer = BloomTokenizerFast.from_pretrained(
            self.base_model,
            add_eos_token=self.add_eos_token
        )  # default add_eos_token=False

        return model, tokenizer

    def tokenize_prompt(self, data_point):
        tokenize_res = self.tokenizer(data_point["input"], truncation=True, padding=False)
        tokenize_res["labels"] = torch.tensor(self.labels.index(data_point["output"]))

        return tokenize_res

    def set_train_data_collator(self):
        self.train_data_collator = transformers.DataCollatorWithPadding(self.tokenizer, return_tensors="pt")

    def finetune(self):
        self.auto_device()
        self.model, self.tokenizer = self.get_model_tokenizer()
        self.set_train_data_collator()

        self.finetune_base()

    def eval_load_model(self):
        self.auto_device()
        self.model, self.tokenizer = self.get_model_tokenizer()
        self.eval_load_model_base()

    def generate(self):
        self.eval_load_model()
        self.generate_base()

    def combine(self):
        self.auto_device()
        self.model, self.tokenizer = self.get_model_tokenizer()
        deloreanized_sd = self.combine_base()
        BloomForSequenceClassification.save_pretrained(
            self.model, self.output_dir, state_dict=deloreanized_sd, max_shard_size=self.max_shard_size
        )

        self.tokenizer.save_pretrained(self.output_dir)


if __name__ == "__main__":
    bloom = BLoomClassify()
    bloom.finetune()
