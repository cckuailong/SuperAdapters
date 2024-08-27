import os
import sys
import re
import copy
import torch
import transformers

from common.base import IGNORE_INDEX
from common.prompt import PROMPT_DICT

from transformers import (
    GemmaTokenizerFast,
    GemmaForCausalLM,
    GenerationConfig,
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


class GemmaSeq2Seq(LLM):
    tokenizer = None

    def get_model_tokenizer(self):
        if self.adapter == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model = GemmaForCausalLM.from_pretrained(
                self.base_model,
                load_in_8bit=self.load_8bit,
                device_map=self.device_map,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
            )
        else:
            model = GemmaForCausalLM.from_pretrained(
                self.base_model,
                load_in_8bit=self.load_8bit,
                device_map=self.device_map,
                low_cpu_mem_usage=True,
            )
        tokenizer = GemmaTokenizerFast.from_pretrained(
            self.base_model,
            add_eos_token=self.add_eos_token
        )  # default add_eos_token=False

        return model, tokenizer

    def generate_prompt(self, data_point):
        # a nasty solution just for now
        if 'Human:' in data_point["instruction"] and 'Assistant:' in data_point["instruction"]:  # TODO
            data_point["instruction"] = data_point["instruction"].replace('Human:', '### Human: ')
            data_point["instruction"] = data_point["instruction"].replace('Assistant:', '### Assistant: ')

            return PROMPT_DICT['prompt_multirun_input'].format_map(data_point)

        prompt_ = PROMPT_DICT['prompt_input'] if data_point["input"] else PROMPT_DICT['prompt_no_input']

        return prompt_.format_map(data_point)

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

    def split_train_data(self, data):
        if self.val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=self.val_set_size, shuffle=True, seed=42
            )
            train_data = (
                train_val["train"].shuffle().map(self.tokenize_prompt)
            )
            val_data = (
                train_val["test"].shuffle().map(self.tokenize_prompt)
            )
        else:
            train_data = data["train"].shuffle().map(self.tokenize_prompt)
            val_data = None

        return train_data, val_data

    def finetune(self, fromdb, iteration):
        self.auto_device()

        if not self.lora_target_modules:
            self.lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj"
            ]

        model, self.tokenizer = self.get_model_tokenizer()
        if self.load_8bit:
            model = prepare_model_for_int8_training(model)

        model = self.load_adapter_config(model)

        data = self.load_train_data(fromdb, iteration)
        print(data)
        if not data:
            print("Warning! Empty Train Data!")
            return

        train_data, val_data = self.split_train_data(data)

        if self.resume_from_checkpoint:
            # Check the available weights and load them
            checkpoint_name = os.path.join(
                self.resume_from_checkpoint, "pytorch_model.bin"
            )  # Full checkpoint
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    self.resume_from_checkpoint, "adapter_model.bin"
                )  # only LoRA model - LoRA config above has to fit
                checkpoint_name_new = os.path.join(
                    self.resume_from_checkpoint, "adapter_model.safetensors"
                )  # when peft >= 0.7.1, the default storage name is "adapter_model.safetensors"
                self.resume_from_checkpoint = (
                    False  # So the trainer won't try loading its state
                )
            # The two files above have a different name depending on how they were saved, but are actually the same.
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
            elif os.path.exists(checkpoint_name_new):
                print(f"Restarting from {checkpoint_name_new}")
                adapters_weights = load_file(checkpoint_name_new)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")

        total_batch_size = self.per_gpu_train_batch_size * self.gradient_accumulation_steps * (self.world_size if self.ddp else 1)
        total_optim_steps = train_data.num_rows // total_batch_size
        saving_step = int(total_optim_steps / 10)
        warmup_steps = int(total_optim_steps / 10)
        train_args = transformers.TrainingArguments(
            per_device_train_batch_size=self.per_gpu_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=warmup_steps if warmup_steps != 1 else 2,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            fp16=self.is_fp16,
            optim="adamw_torch",
            logging_steps=self.logging_steps,
            evaluation_strategy="steps" if self.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=saving_step if self.val_set_size > 0 else None,
            save_steps=saving_step,
            # max_steps=200,
            output_dir=self.output_dir,
            save_total_limit=11,
            load_best_model_at_end=True if self.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if self.ddp else None,
            group_by_length=self.group_by_length,
            use_mps_device=self.use_mps_device,
            report_to=None if self.disable_wandb else "wandb"
        )

        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(self.tokenizer, return_tensors="pt", padding=True),
        )

        model.config.use_cache = False

        if torch.__version__ >= "2" and sys.platform != "win32":
            try:
                model = torch.compile(model)
            except:
                print("Warning: torch.compile() failed, will skip it.")

        trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)

        model.save_pretrained(self.output_dir)

        print("\n If there's a warning about missing keys above, please disregard :)")

    def generate_eval_prompt(self, instruction, input=None):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:"""

    def evaluate(self,
                 model,
                 instruction,
                 input=None,
                 **kwargs,
                 ):
        prompt = self.generate_eval_prompt(instruction, input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=4,
            do_sample=True,
            no_repeat_ngram_size=6,
            repetition_penalty=1.8,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=self.max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)

        return output.split("### Response:")[1].strip()

    def load_model(self):
        self.auto_device()

        model, self.tokenizer = self.get_model_tokenizer()

        if self.adapter_weights != "None":
            model = PeftModel.from_pretrained(
                model,
                self.adapter_weights,
            )

        if not self.load_8bit and self.device != "cpu":
            model.half()

        if self.load_8bit:
            model.eval()
        else:
            model.to(self.device).eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            try:
                model = torch.compile(model)
            except:
                print("Warning: torch.compile() failed, will skip it.")

        return model

    def generate(self, instruction, input, data, fromdb, type, iteration, test_iteration, max_input):
        model = self.load_model()

        eval_inputs = self.get_eval_input(instruction, input, data, fromdb, type, iteration, max_input)

        for item in tqdm(eval_inputs):
            try:
                response = self.evaluate(model, item["instruction"], item["input"])
                if response[-4:] == "</s>":
                    response = response[:-4]
                elif response[-15:] == "<|end_of_text|>":
                    response = response[:-15]
            except Exception as e:
                if self.debug:
                    print("[DEBUG] Error: " + str(e))
                response = "Eval Error"

            item["ac_output"] = response

        if self.web:
            return eval_inputs[0]["ac_output"]
        else:
            self.eval_output(eval_inputs, data, fromdb, type, iteration, test_iteration)


if __name__ == "__main__":
    gemma = GemmaSeq2Seq()
    gemma.finetune()
