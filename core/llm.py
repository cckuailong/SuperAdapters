import os
import sys
import json
import torch
import transformers
from transformers import (
    GenerationConfig,
)
from safetensors.torch import load_file
from peft import (
    AdaLoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel,
)

from typing import List
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from common.prompt import PROMPT_DICT

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class LLM:
    # system
    debug: bool = False
    web: bool = False

    # base params
    base_model: str = "LLMs/chatglm/chatglm-6b"
    model_type: str = "chatglm"
    task_type: str = "seq2seq"
    data_path: str = "data/train"
    labels: list = ["0", "1"]
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
    max_input: int = 0

    model = None
    tokenizer = None
    tokenize_prompt = None
    train_data_collator = None

    fromdb = False
    db_type = ""
    db_iteration = ""
    db_test_iteration = ""
    db_item_num: int = 0
    instruction = ""
    input = ""
    test_data_path = ""



    def load_adapter_config(self, model):
        if self.task_type == "seq2seq":
            t_type = TaskType.CAUSAL_LM
        elif self.task_type == "classify":
            t_type = TaskType.SEQ_CLS
        else:
            t_type = TaskType.CAUSAL_LM
        if self.adapter == "lora" or self.adapter == "qlora":
            config = LoraConfig(
                task_type=t_type,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                bias="none",
                inference_mode=False,
            )
        elif self.adapter == 'adalora':
            config = AdaLoraConfig(
                task_type=t_type,
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
                task_type=t_type,
                num_virtual_tokens=self.num_virtual_tokens,
                encoder_hidden_size=self.mapping_hidden_dim,
                prefix_projection=True
            )
        elif self.adapter == "p_tuning":
            config = PromptEncoderConfig(
                task_type=t_type,
                num_virtual_tokens=self.num_virtual_tokens,
                encoder_hidden_size=self.mapping_hidden_dim
            )
        elif self.adapter == "prompt":
            config = PromptTuningConfig(
                task_type=t_type,
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

    # -------------- Inference ----------------
    #
    #
    # Generate the prompt to format the LLM input.
    def generate_prompt(self, data_point):
        # a nasty solution just for now
        if 'Human:' in data_point["instruction"] and 'Assistant:' in data_point["instruction"]:  # TODO
            data_point["instruction"] = data_point["instruction"].replace('Human:', '### Human: ')
            data_point["instruction"] = data_point["instruction"].replace('Assistant:', '### Assistant: ')

            return PROMPT_DICT['prompt_multirun_input'].format_map(data_point)

        prompt_ = PROMPT_DICT['prompt_input'] if data_point["input"] else PROMPT_DICT['prompt_no_input']

        return prompt_.format_map(data_point)

    # Load the train data
    def load_train_data(self, fromdb, s_iteration):
        data = None
        if fromdb:
            train_data_set = []

            from common.db import get_mysql_conn
            conn = get_mysql_conn()
            cur = conn.cursor()
            sql = "select instruction,input,output from playbooks_all where iteration=%s and `type`='train' and is_check > 0"
            if self.db_item_num > 0:
                sql += " limit {}".format(self.db_item_num)
            cur.execute(sql, s_iteration)
            items = cur.fetchall()
            cur.close()
            conn.close()
            for item in items:
                instruction, input, output = item
                train_data_set.append({
                    "instruction": instruction,
                    "input": input,
                    "output": output
                })
            data = DatasetDict({"train": Dataset.from_list(train_data_set)})
        elif self.data_path:
            if self.data_path.endswith(".json") or self.data_path.endswith(".jsonl"):
                data = load_dataset("json", data_files=self.data_path)
            else:
                data = load_dataset(self.data_path)

        return data

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

    # Finetune main function.
    def finetune_base(self):
        if self.load_8bit:
            self.model = prepare_model_for_int8_training(self.model)

        self.model = self.load_adapter_config(self.model)

        data = self.load_train_data(self.fromdb, self.db_iteration)
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
                set_peft_model_state_dict(self.model, adapters_weights)
            elif os.path.exists(checkpoint_name_new):
                print(f"Restarting from {checkpoint_name_new}")
                adapters_weights = load_file(checkpoint_name_new)
                set_peft_model_state_dict(self.model, adapters_weights)
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
            eval_strategy="steps" if self.val_set_size > 0 else "no",
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
            model=self.model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=train_args,
            data_collator=self.train_data_collator,
        )

        self.model.config.use_cache = False

        if torch.__version__ >= "2" and sys.platform != "win32":
            try:
                self.model = torch.compile(self.model)
            except:
                print("Warning: torch.compile() failed, will skip it.")

        trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)

        self.model.save_pretrained(self.output_dir)

        print("\n If there's a warning about missing keys above, please disregard :)")

    # -------------- Inference ----------------
    #
    #
    # Function evaluate() is the core.
    #
    # Generate the prompt to format the LLM output.
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

    # Inference main function.
    def evaluate(self,
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
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=self.max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)

        return output.split("### Response:")[1].strip()

    # -------------- Format Inference - ---------------
    # Get Input or the question.
    def get_eval_input(self):
        result = []
        if self.fromdb:
            from common.db import get_mysql_conn
            conn = get_mysql_conn()
            cur = conn.cursor()
            sql = "select payload_uuid,instruction,input,output from playbooks where type=%s and iteration=%s"
            if self.db_item_num > 0:
                sql += " limit {}".format(self.db_item_num)
            cur.execute(sql, (self.db_type, self.db_iteration))
            items = cur.fetchall()
            cur.close()
            conn.close()

            for item in items:
                payload_uuid, instruction, input, output = item
                if self.max_input:
                    input = input[:self.max_input]
                result.append({
                    "payload_uuid": payload_uuid,
                    "instruction": instruction,
                    "input": input,
                    "output": output
                })
        elif self.test_data_path:
            with open(self.test_data_path, "r") as f:
                test_items = json.loads(f.read())
            result = test_items
        else:
            if self.max_input:
                self.input = self.input[:self.max_input]
            result.append({
                "instruction": self.instruction,
                "input": self.input
            })

        print("Find {} cases".format(len(result)))

        return result

    # Help model to adapt some situations.
    def eval_load_model_base(self):
        if self.adapter_weights != "None":
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_weights,
            )

        if not self.load_8bit and self.device != "cpu":
            self.model.half()


        if self.load_8bit:
            self.model.eval()
        else:
            self.model.to(self.device).eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            try:
                self.model = torch.compile(self.model)
            except:
                print("Warning: torch.compile() failed, will skip it.")

    # Format the output of the LLM.
    def eval_output(self, eval_inputs):
        if self.fromdb:
            data_set = []
            for item in eval_inputs:
                data_set.append((item["payload_uuid"], self.db_type, item["instruction"], item["input"], item["output"],
                                 item["ac_output"], self.db_iteration, self.db_test_iteration))

            from common.db import get_mysql_conn
            conn = get_mysql_conn()
            cur = conn.cursor()
            sql = "insert into result (payload_uuid,type,instruction,input,output,ac_output,iteration,test_iteration) values(%s,%s,%s,%s,%s,%s,%s,%s)"
            cur.executemany(sql, data_set)
            conn.commit()
            cur.close()
            conn.close()

            print("Finish eval!")
        elif self.test_data_path:
            case_cnt = 0
            for item in eval_inputs:
                case_cnt += 1
                print("[*] Case: {}\n--------\nExpect: \n{}\n----------------\nOutput: \n{}\n".format(case_cnt, item["output"], item["ac_output"]))
        else:
            print("LLM says: \n{}".format(eval_inputs[0]["ac_output"]))

    # Format Inference main function.
    def generate_base(self):
        eval_inputs = self.get_eval_input()

        for item in tqdm(eval_inputs):
            try:
                response = self.evaluate(item["instruction"], item["input"])
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
            self.eval_output(eval_inputs)