import sys
import argparse

from core.seq2seq.chatglm import ChatGLMSeq2Seq
from core.seq2seq.llama import LLAMASeq2Seq
from core.seq2seq.bloom import BLoomSeq2Seq
from core.seq2seq.qwen import QwenSeq2Seq

from core.classify.llama import LLAMAClassify
from core.classify.bloom import BLoomClassify


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for all.')

    # base
    parser.add_argument('--instruction', default="Hello", type=str)
    parser.add_argument('--input', default=None, type=str)
    parser.add_argument('--data', default=None, help="The DIR of test data", type=str)
    parser.add_argument('--model_type', default="llama", choices=['llama', 'chatglm', 'chatglm2', 'bloom', 'qwen'])
    parser.add_argument('--task_type', default="seq2seq", choices=['seq2seq', 'classify'])
    parser.add_argument('--labels', default="[\"0\", \"1\"]",
                        help="Labels to classify, only used when task_type is classify")
    parser.add_argument('--model_path', default="LLMs/open-llama/openllama-3b", type=str)
    parser.add_argument('--adapter_weights', default="None", type=str, help="The DIR of adapter weights")

    parser.add_argument('--load_8bit', action="store_true")

    # generate
    parser.add_argument('--temperature', default="0.7", type=float, help="temperature higher, LLM is more creative")
    parser.add_argument('--top_p', default="0.9", type=float)
    parser.add_argument('--top_k', default="40", type=int)
    parser.add_argument('--max_new_tokens', default="512", type=int)

    # fromdb
    parser.add_argument('--fromdb', action="store_true")
    parser.add_argument('--db_type', default=None, type=str, help="The record is whether 'train' or 'test'.")
    parser.add_argument('--db_iteration', default=None, type=str, help="The record's set name.")
    parser.add_argument('--db_test_iteration', default=None, type=str, help="The record's test set name.")


    args, _ = parser.parse_known_args()

    if args.task_type == "seq2seq":
        if args.model_type == "chatglm" or args.model_type == "chatglm2":
            llm = ChatGLMSeq2Seq()
        elif args.model_type == "llama":
            llm = LLAMASeq2Seq()
        elif args.model_type == "bloom":
            llm = BLoomSeq2Seq()
        elif args.model_type == "qwen":
            llm = QwenSeq2Seq()
        else:
            print("model_type should be llama/bloom/chatglm/chatglm2")
            sys.exit(-1)
    elif args.task_type == "classify":
        if args.model_type == "chatglm" or args.model_type == "chatglm2":
            print("Classify with ChatGLM is not support now.")
            sys.exit(-1)
        elif args.model_type == "llama":
            llm = LLAMAClassify()
        elif args.model_type == "bloom":
            llm = BLoomClassify()
        elif args.model_type == "qwen":
            print("Classify with Qwen is not support now.")
            sys.exit(-1)
        else:
            print("model_type should be llama/bloom/chatglm/chatglm2")
            sys.exit(-1)

    llm.base_model = args.model_path
    llm.adapter_weights = args.adapter_weights

    llm.load_8bit = args.load_8bit

    llm.temperature = args.temperature
    llm.top_p = args.top_p
    llm.top_k = args.top_k
    llm.max_new_tokens = args.max_new_tokens

    llm.generate(args.instruction, args.input, args.data, args.fromdb, args.db_type, args.db_iteration, args.db_test_iteration)

