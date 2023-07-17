import argparse
from core.chatglm import ChatGLM
from core.llama import LLAMA
from core.bloom import BLoom


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    # base
    parser.add_argument('--instruction', default="Hello", type=str)
    parser.add_argument('--input', default=None, type=str)
    parser.add_argument('--data', default=None, help="The DIR of test data", type=str)
    parser.add_argument('--model_type', default="llama", choices=['llama', 'chatglm', 'chatglm2', 'bloom'])
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
    parser.add_argument('--db_type', default=None, type=str)
    parser.add_argument('--db_iteration', default=None, type=str)
    parser.add_argument('--db_test_iteration', default=None, type=str)


    args, _ = parser.parse_known_args()

    if args.model_type == "chatglm" or args.model_type == "chatglm2":
        llm = ChatGLM()
    elif args.model_type == "llama":
        llm = LLAMA()
    elif args.model_type == "bloom":
        llm = BLoom()

    llm.base_model = args.model_path
    llm.adapter_weights = args.adapter_weights

    llm.load_8bit = args.load_8bit

    llm.temperature = args.temperature
    llm.top_p = args.top_p
    llm.top_k = args.top_k
    llm.max_new_tokens = args.max_new_tokens

    llm.generate(args.instruction, args.input, args.data, args.fromdb, args.db_type, args.db_iteration, args.db_test_iteration)

