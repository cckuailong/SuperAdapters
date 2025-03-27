import re
import copy
import torch
import transformers

from common.base import IGNORE_INDEX
from common.prompt import PROMPT_DICT

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import importlib
if importlib.util.find_spec('unsloth') is not None:
    from unsloth import FastLanguageModel

from core.llm import LLM


class LLAMASeq2Seq(LLM):
    def __init__(self):
        if not self.lora_target_modules:
            if self.model_type == "llama2" or self.model_type == "llama3":
                self.lora_target_modules = [
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "down_proj",
                    "up_proj"
                ]
            else:
                self.lora_target_modules = [
                    "q_proj",
                    "v_proj"
                ]

    def get_model_tokenizer(self):
        if self.adapter == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            if self.model_type == "llama3":
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    load_in_8bit=self.load_8bit,
                    device_map=self.device_map,
                    low_cpu_mem_usage=True,
                    quantization_config=bnb_config,
                    trust_remote_code=True
                )
            else:
                model = LlamaForCausalLM.from_pretrained(
                    self.base_model,
                    load_in_8bit=self.load_8bit,
                    device_map=self.device_map,
                    low_cpu_mem_usage=True,
                    quantization_config=bnb_config
                )
        else:
            if self.model_type == "llama3":
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    load_in_8bit=self.load_8bit,
                    device_map=self.device_map,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                model = LlamaForCausalLM.from_pretrained(
                    self.base_model,
                    load_in_8bit=self.load_8bit,
                    device_map=self.device_map,
                    low_cpu_mem_usage=True
                )
        if self.model_type == "llama3":
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                add_eos_token=self.add_eos_token,
                trust_remote_code=True
            )
        else:
            tokenizer = LlamaTokenizer.from_pretrained(
                self.base_model,
                add_eos_token=self.add_eos_token
            )

        # Some Models do not have pad_token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

        return model, tokenizer

    def get_model_tokenizer_unsloth(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_new_tokens,
            dtype=None,
            trust_remote_code=True,
            load_in_4bit=True if self.adapter == "qlora" else False,
        )
        if self.adapter in ['lora', 'qlora']:
            target_modules = find_all_linear_names(model, args.train_mode)
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_rank,
                target_modules=target_modules,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                use_gradient_checkpointing=True,
                random_state=training_args.seed,
                max_seq_length=args.max_seq_length,
            )
            logger.info(f'target_modules: {target_modules}')
        return {
            'model': model,
            'ref_model': None,
            'peft_config': None
        }

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
        if not self.vllm:
            self.eval_load_model()
        self.generate_base()

    def combine(self):
        self.device = "cpu"
        self.auto_device()
        if self.model_type == "llama3":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map=self.device_map
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                add_eos_token=self.add_eos_token,
                trust_remote_code=True
            )
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                self.base_model,
                device_map=self.device_map,
                torch_dtype=torch.float16,
            )
            self.tokenizer = LlamaTokenizer.from_pretrained(
                self.base_model,
                add_eos_token=self.add_eos_token
            )
        deloreanized_sd = self.combine_base()
        LlamaForCausalLM.save_pretrained(
            self.model, self.output_dir, state_dict=deloreanized_sd, max_shard_size=self.max_shard_size
        )

        self.tokenizer.save_pretrained(self.output_dir)


if __name__ == "__main__":
    llama = LLAMASeq2Seq()
    llama.finetune()
