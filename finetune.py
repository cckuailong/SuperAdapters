import os
import sys
import argparse

from core.chatglm import ChatGLM
from core.llama import LLAMA
from core.bloom import BLoom


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    # base
    parser.add_argument('--data', type=str, default="data/train/", help='the data used for instructing tuning')
    parser.add_argument('--model_type', default="llama", choices=['llama', 'chatglm', 'chatglm2', 'bloom'])
    parser.add_argument('--model_path', default="LLMs/open-llama/openllama-3b", type=str)
    parser.add_argument('--output_dir', default="output/", type=str, help="The DIR to save the model")
    parser.add_argument('--disable_wandb', action="store_true", help="Disable report to wandb")

    # adapter
    parser.add_argument('--adapter', default="lora", choices=['lora', 'adalora', 'prompt', 'p_tuning', 'prefix'])
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    parser.add_argument('--lora_target_modules', nargs='+',
                        help="the module to be injected, e.g. q_proj/v_proj/k_proj/o_proj for llama, query_key_value for bloom&GLM",
                        default=None)
    parser.add_argument('--adalora_init_r', default=12, type=int)
    parser.add_argument("--adalora_tinit", type=int, default=200,
                        help="number of warmup steps for AdaLoRA wherein no pruning is performed")
    parser.add_argument("--adalora_tfinal", type=int, default=1000,
                        help=" fix the resulting budget distribution and fine-tune the model for tfinal steps when using AdaLoRA ")
    parser.add_argument("--adalora_delta_t", type=int, default=10, help="interval of steps for AdaLoRA to update rank")
    parser.add_argument('--num_virtual_tokens', default=20, type=int)
    parser.add_argument('--mapping_hidden_dim', default=128, type=int)

    # train
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--cutoff_len', default=512, type=int)
    parser.add_argument('--val_set_size', default=0.2, type=float)
    parser.add_argument('--group_by_length', action="store_true")
    parser.add_argument('--logging_steps', default=20, type=int)

    parser.add_argument('--load_8bit', action="store_true")
    parser.add_argument('--add_eos_token', action="store_true")
    parser.add_argument('--resume_from_checkpoint', nargs='?', default=None, const=True,
                        help='resume from the specified or the latest checkpoint, e.g. `--resume_from_checkpoint [path]` or `--resume_from_checkpoint`')
    parser.add_argument('--per_gpu_train_batch_size', default=4, type=int, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--gradient_accumulation_steps', default=32, type=int)

    parser.add_argument('--fromdb', action="store_true")
    parser.add_argument('--db_iteration', default=None, type=str, help="The record's set name.")

    args, _ = parser.parse_known_args()

    if args.model_type == "chatglm" or args.model_type == "chatglm2":
        llm = ChatGLM()
    elif args.model_type == "llama":
        llm = LLAMA()
    elif args.model_type == "bloom":
        llm = BLoom()
    else:
        print("model_type should be llama/bloom/chatglm/chatglm2")
        sys.exit(-1)

    llm.data_path = args.data
    llm.model_type = args.model_type
    llm.base_model = args.model_path
    llm.output_dir = args.output_dir
    llm.disable_wandb = args.disable_wandb
    if llm.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    llm.adapter = args.adapter
    llm.lora_r = args.lora_r
    llm.lora_alpha = args.lora_alpha
    llm.lora_dropout = args.lora_dropout
    llm.lora_target_modules = args.lora_target_modules
    llm.adalora_init_r = args.adalora_init_r
    llm.adalora_tinit = args.adalora_tinit
    llm.adalora_tfinal = args.adalora_tfinal
    llm.adalora_delta_t = args.adalora_delta_t

    llm.num_virtual_tokens = args.num_virtual_tokens
    llm.mapping_hidden_dim = args.mapping_hidden_dim
    llm.epochs = args.epochs
    llm.learning_rate = args.learning_rate
    llm.cutoff_len = args.cutoff_len
    llm.val_set_size = args.val_set_size
    llm.group_by_length = args.group_by_length
    llm.logging_steps = args.logging_steps

    llm.load_8bit = args.load_8bit
    llm.add_eos_token = args.add_eos_token
    llm.resume_from_checkpoint = args.resume_from_checkpoint
    llm.per_gpu_train_batch_size = args.per_gpu_train_batch_size
    llm.gradient_accumulation_steps = args.gradient_accumulation_steps

    if not os.path.exists(llm.output_dir):
        os.makedirs(llm.output_dir)
        print("Warning: Directory {} Not Found, create automatically")

    llm.finetune(args.fromdb, args.db_iteration)
