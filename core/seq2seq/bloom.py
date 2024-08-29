import re
import copy
import torch
import transformers

from common.base import IGNORE_INDEX
from common.prompt import PROMPT_DICT

from transformers import (
    BloomTokenizerFast,
    BloomForCausalLM,
    BitsAndBytesConfig
)

from core.llm import LLM


class BLoomSeq2Seq(LLM):
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
            model = BloomForCausalLM.from_pretrained(
                self.base_model,
                load_in_8bit=self.load_8bit,
                device_map=self.device_map,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
            )
        else:
            model = BloomForCausalLM.from_pretrained(
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

    def tokenize(self, prompt):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            #    padding="max_length",
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
            "labels": copy.deepcopy(result["input_ids"])
        }

    def tokenize_prompt(self, data_point):
        prompt_no_resp = self.generate_prompt(data_point)

        if 'multi-round dialogue' in prompt_no_resp:
            prompt_no_resp = re.sub(r'(?<!\n)\n### ', '\n</s>### ', prompt_no_resp)
            prompt_no_resp += '</s>'
            """ so far the prompt_no_resp looks like:
            Below is an multi-round dialogue ...
            ### Human: ...
            </s>### Assistant: ...
            </s>### Human: ...
            ...
            </s>### Assistant: ... </s>
            """

            inputs_with_offsets = self.tokenizer(prompt_no_resp, return_offsets_mapping=True)
            labels = copy.deepcopy(inputs_with_offsets['input_ids'])
            source_len = len(self.tokenizer(PROMPT_DICT['prompt_multirun_input'].split('\n\n')[0] + '\n\n')['input_ids'])
            labels[:source_len] = [IGNORE_INDEX] * source_len
            offsets = inputs_with_offsets["offset_mapping"]

            matches = re.finditer(r'### (?!Assistant:)(.*?)</s>', prompt_no_resp, re.DOTALL)

            for match in matches:
                start_pos, end_pos = match.span()
                start_idx = None
                end_idx = None

                for i, (start, end) in enumerate(offsets):
                    if start <= start_pos < end:
                        start_idx = i
                    if start <= end_pos < end:
                        end_idx = i

                if start_idx is not None and end_idx is not None:
                    for i in range(start_idx, end_idx - 1):
                        labels[i] = IGNORE_INDEX

            return dict(
                input_ids=inputs_with_offsets['input_ids'],
                attention_mask=inputs_with_offsets['attention_mask'],
                labels=labels,
            )
        else:
            tokenized_result = self.tokenize(prompt_no_resp)

            source_len = len(tokenized_result['input_ids'])
            prompt_with_response = prompt_no_resp + " " + data_point["output"]
            prompt_with_response += " " + self.tokenizer.eos_token

            tokenized_with_response = self.tokenize(prompt_with_response)

            tokenized_with_response["labels"] = [IGNORE_INDEX] * source_len + tokenized_with_response["labels"][source_len:]

            return tokenized_with_response

    def set_train_data_collator(self):
        self.train_data_collator = transformers.DataCollatorForSeq2Seq(self.tokenizer, return_tensors="pt", padding=True)

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


if __name__ == "__main__":
    bloom = BLoomSeq2Seq()
    bloom.finetune()
