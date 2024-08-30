import re
import copy
import torch
from typing import Dict, Optional, Sequence, Union

from common.base import IGNORE_INDEX
from common.prompt import PROMPT_DICT

from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    BatchEncoding,
    BitsAndBytesConfig
)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from core.llm import LLM


class ChatGLMCollator(DataCollatorWithPadding):
    r"""
    Data collator for ChatGLM. It is capable of dynamically padding for batched data.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            ignore_pad_token_for_loss: Optional[bool] = False,
            use_v2: Optional[bool] = False
    ):
        super().__init__(tokenizer, padding=True)
        self.model = model
        self.label_pad_token_id = IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id
        if use_v2:
            self.get_attention_masks = self.get_attention_masks_v2
            self.get_position_ids = self.get_position_ids_v2
        else:
            self.get_attention_masks = self.get_attention_masks_v1
            self.get_position_ids = self.get_position_ids_v1

    def get_attention_masks_v1(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        r"""
        Generates attention masks for left-padded sequences.

        Note that ChatGLM assigns False on token to be attended in attention mask. In general settings, it should be True.

        According to: https://huggingface.co/THUDM/chatglm-6b/blob/v1.1.0/modeling_chatglm.py#L680
        """
        batch_size, seq_length = input_ids.size()
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
        attention_mask.tril_()

        for i, seq in enumerate(input_ids):
            attention_mask[i, :, :(seq == self.tokenizer.bos_token_id).nonzero()[0].item()] = 1  # context
            attention_mask[i, :, :(seq != self.tokenizer.pad_token_id).nonzero()[0].item()] = 0  # padding

        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()
        return attention_mask

    def get_position_ids_v1(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        r"""
        Generates position ids for left-padded sequenes.

        According to: https://huggingface.co/THUDM/chatglm-6b/blob/v1.1.0/modeling_chatglm.py#L692
        """
        batch_size, seq_length = input_ids.size()
        mask: int = self.model.config.mask_token_id
        gmask: int = self.model.config.gmask_token_id
        position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)
        block_position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)

        for i, seq in enumerate(input_ids):
            mask_token = gmask if gmask in seq else mask
            context_length = (seq == self.tokenizer.bos_token_id).nonzero()[0].item()
            padding_length = (seq != self.tokenizer.pad_token_id).nonzero()[0].item()
            position_ids[i, padding_length:] = torch.arange(
                seq_length - padding_length,
                dtype=torch.long,
                device=device
            )
            if self.model.position_encoding_2d or (mask_token != gmask):  # 2d position encoding or not gMASK
                position_ids[i, context_length:] = (seq == mask_token).nonzero()[
                                                       0].item() - padding_length  # mask position
            block_position_ids[i, context_length:] = torch.arange(
                seq_length - context_length,
                dtype=torch.long,
                device=device
            ) + 1

        if self.model.position_encoding_2d:
            position_ids = torch.stack((position_ids, block_position_ids), dim=1)

        return position_ids

    def get_attention_masks_v2(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        r"""
        Generates attention masks for left-padded sequences.
        """
        batch_size, seq_length = input_ids.size()
        attention_mask = torch.ones((batch_size, seq_length), device=device)

        for i, seq in enumerate(input_ids):
            attention_mask[i, :(seq != self.tokenizer.pad_token_id).nonzero()[0].item()] = 0  # padding

        return attention_mask

    def get_position_ids_v2(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        r"""
        Generates position ids for left-padded sequenes.
        """
        batch_size, seq_length = input_ids.size()
        position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)

        for i, seq in enumerate(input_ids):
            padding_length = (seq != self.tokenizer.pad_token_id).nonzero()[0].item()
            position_ids[i, padding_length:] = torch.arange(seq_length - padding_length, dtype=torch.long,
                                                            device=device)

        return position_ids

    def __call__(self, features: Sequence[Dict[str, Union[torch.Tensor, Sequence[int]]]]) -> BatchEncoding:
        r"""
        Pads batched data to the longest sequence in the batch.

        We adopt left-padding in both training and evaluation.
        """
        if isinstance(features[0]["input_ids"], torch.Tensor):
            input_ids = [feature["input_ids"].clone().detach().flip(0) for feature in features]
        else:
            input_ids = [torch.tensor(feature["input_ids"]).flip(0) for feature in features]

        if "labels" in features[0]:
            if isinstance(features[0]["labels"], torch.Tensor):
                labels = [feature["labels"].clone().detach().flip(0) for feature in features]
            else:
                labels = [torch.tensor(feature["labels"]).flip(0) for feature in features]
            input_ids = input_ids + labels  # pad them to the same length

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        ).flip(-1)

        batch = {}

        if "labels" in features[0]:
            input_ids, labels = input_ids.split(len(features), dim=0)
            labels = torch.where(labels != self.tokenizer.pad_token_id, labels, self.label_pad_token_id)
            batch["labels"] = labels

        batch["input_ids"] = input_ids
        batch["attention_mask"] = self.get_attention_masks(input_ids, device=input_ids.device)
        batch["position_ids"] = self.get_position_ids(input_ids, device=input_ids.device)

        return BatchEncoding(batch)


class ChatGLMSeq2Seq(LLM):
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
            model = AutoModel.from_pretrained(
                self.base_model,
                load_in_8bit=self.load_8bit,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map=self.device_map,
                quantization_config=bnb_config
            )
        else:
            model = AutoModel.from_pretrained(
                self.base_model,
                load_in_8bit=self.load_8bit,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map=self.device_map,
            )
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            add_eos_token=self.add_eos_token
        )  # default add_eos_token=False

        return model, tokenizer

    def prompt_tokenize(self, prompt):
        input_ids = self.tokenizer.encode(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            #    padding="max_length",
            padding=False,
        )
        return {
            "input_ids": input_ids,
            "labels": copy.deepcopy(input_ids)
        }

    def completion_tokenize(self, completion):
        input_ids = self.tokenizer.encode(
            completion,
            truncation=True,
            max_length=self.cutoff_len,
            # add_special_tokens=False
        )
        return {
            "input_ids": input_ids,
            "labels": copy.deepcopy(input_ids)
        }

    def tokenize_prompt(self, data_point):
        prompt_no_resp = self.generate_prompt(data_point)

        if 'multi-round dialogue' in prompt_no_resp:
            inputs_with_offsets = self.tokenizer(prompt_no_resp, return_offsets_mapping=True)
            labels = copy.deepcopy(inputs_with_offsets['input_ids'])
            source_len = len(
                self.tokenizer(PROMPT_DICT['prompt_multirun_input'].split('\n\n')[0] + '\n\n')['input_ids'])
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
            tokenized_result = self.prompt_tokenize(prompt_no_resp)

            source_len = len(tokenized_result['input_ids'])
            prompt_with_response = prompt_no_resp + " " + data_point["output"]
            prompt_with_response += " " + self.tokenizer.eos_token

            tokenized_with_response = self.completion_tokenize(prompt_with_response)
            tokenized_with_response["input_ids"] = tokenized_result['input_ids'] + tokenized_with_response["input_ids"][
                                                                                   source_len - 2:-2]
            tokenized_with_response["labels"] = tokenized_result['labels'] + tokenized_with_response["labels"][
                                                                             source_len - 2:-2]

            tokenized_with_response["labels"] = [IGNORE_INDEX] * source_len + tokenized_with_response["labels"][
                                                                              source_len:]

            return tokenized_with_response

    def set_train_data_collator(self):
        self.train_data_collator = ChatGLMCollator(
                self.tokenizer,
                model=self.model,
                ignore_pad_token_for_loss=False,
                use_v2=True if self.model_type == "chatglm2" else False
            )

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
        self.model, self.tokenizer = self.get_model_tokenizer()
        deloreanized_sd = self.combine_base()
        AutoModel.save_pretrained(
            self.model, self.output_dir, state_dict=deloreanized_sd, max_shard_size=self.max_shard_size
        )

        self.tokenizer.save_pretrained(self.output_dir)


if __name__ == "__main__":
    chatglm = ChatGLMSeq2Seq()
    chatglm.finetune()
