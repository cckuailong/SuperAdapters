import os
import sys
import argparse

from core.seq2seq.chatglm import ChatGLMSeq2Seq
from core.seq2seq.llama import LLAMASeq2Seq
from core.seq2seq.bloom import BLoomSeq2Seq
from core.seq2seq.qwen import QwenSeq2Seq
from core.seq2seq.baichuan import BaichuanSeq2Seq
from core.seq2seq.mixtral import MixtralSeq2Seq
from core.seq2seq.phi import PhiSeq2Seq
from core.seq2seq.gemma import GemmaSeq2Seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for all.')

    subparsers = parser.add_subparsers(dest='tool', help='Tools or Modules')
    parser_combine = subparsers.add_parser('combine', help='Combine Base model weight and Adapter weight')
    parser_combine.add_argument('--model_type', default="llama",
                        choices=['llama', 'llama2', 'llama3', 'chatglm', 'chatglm2', 'bloom', 'qwen', "baichuan",
                                 "mixtral", "phi", "phi3", "gemma"])
    parser_combine.add_argument('--model_path', default=None, type=str)
    parser_combine.add_argument('--adapter_weights', default=None, type=str, help="The DIR of adapter weights")
    parser_combine.add_argument('--output_dir', default="combined_model/", type=str, help="The DIR to save the model")
    parser_combine.add_argument('--max_shard_size', default="5GB", type=str, help="Max size of each of the combined model weight, like 1GB,5GB,etc.")

    args, _ = parser.parse_known_args()

    if args.tool == 'combine':
        if not args.model_path or not args.adapter_weights:
            print("[Error] Miss param: model_path or adapter_weights")
            sys.exit(-1)
        if args.model_type == "chatglm" or args.model_type == "chatglm2":
            llm = ChatGLMSeq2Seq()
        elif args.model_type == "llama" or args.model_type == "llama2" or args.model_type == "llama3":
            llm = LLAMASeq2Seq()
        elif args.model_type == "bloom":
            llm = BLoomSeq2Seq()
        elif args.model_type == "qwen":
            llm = QwenSeq2Seq()
        elif args.model_type == "baichuan":
            llm = BaichuanSeq2Seq()
        elif args.model_type == "mixtral":
            llm = MixtralSeq2Seq()
        elif args.model_type == "phi" or args.model_type == "phi3":
            llm = PhiSeq2Seq()
        elif args.model_type == "gemma":
            llm = GemmaSeq2Seq()
        else:
            print("model_type should be llama/llama2/llama3/bloom/chatglm/chatglm2/qwen/baichuan/mixtral/phi/phi3/gemma")
            sys.exit(-1)

        llm.base_model = args.model_path
        llm.model_type = args.model_type
        llm.adapter_weights = args.adapter_weights
        llm.output_dir = args.output_dir
        llm.max_shard_size = args.max_shard_size

        if not os.path.exists(llm.output_dir):
            os.makedirs(llm.output_dir)
            print("Warning: Directory {} Not Found, create automatically".format(llm.output_dir))

        llm.combine()
        print("[Success] Combine Adapter weight, new model is in {}".format(llm.output_dir))

