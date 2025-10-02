from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from accelerate.utils.modeling import get_balanced_memory
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model
from torch import nn
import tqdm
import os
import sys
import hf_models
from statlog import Log
from datetime import datetime
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path
from time import sleep


# function for printing the usage text
parser = argparse.ArgumentParser(
    description="Run LLM tests with configurable options."
)

parser.add_argument("--model_path", type=str,
                    help="Path of the hf model")

parser.add_argument("--inputfile", type=str, default="./input.txt",
                    help="File used as input for text generation (Default: ./input.txt)")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--run_tasks", action="store_true", help="Run the tasks workflow")
group.add_argument("--run_input", action="store_true", help="Run the input workflow")


parser.add_argument("--iterations", type=int, default=5,
                    help="Number of iterations to repeat individual tests (Default: 5)")

parser.add_argument("--outputdir", type=str, default="./out",
                    help="Directory to output log files (Default: ./out)")

parser.add_argument("--tokens", type=int, default=64,
                    help="Number of tokens to generate (Default: 64)")

# New options (quantization / training)
parser.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16"])

parser.add_argument("--batch_size", type=int, default=1,
                    help="Batch size")

parser.add_argument("--tasks", type=str, default=None,
                    help="Tasks to evaluate")

parser.add_argument("--num_fewshot", type=int, default=0)

parser.add_argument("--auto_parallel", action="store_true",
                    help="Automatically set parallel and batch_size")

parser.add_argument("--w_bit", type=int, default=None,
                    help="Quantization bit width for weights")

parser.add_argument("--q_group_size", type=int, default=-1,
                    help="Group size for quantization (Default: -1)")

parser.add_argument("--no_zero_point", action="store_true",
                    help="Disable zero_point")

parser.add_argument("--q_backend", type=str, default="fake",
                    choices=["fake", "real"],
                    help="Quantization backend")

# Save/load quantized weights
parser.add_argument("--dump_quant", type=str, default=None,
                    help="Save quantized model")

parser.add_argument("--dump_fake", type=str, default=None,
                    help="Save fake-quantized model")

parser.add_argument("--load_quant", type=str, default=None,
                    help="Load quantized model")

# Apply/save/load AWQ
parser.add_argument("--run_awq", action="store_true",
                    help="Perform AWQ search process")

parser.add_argument("--dump_awq", type=str, default=None,
                    help="Save the AWQ search results")

parser.add_argument("--load_awq", type=str, default=None,
                    help="Load the AWQ search results")

args = parser.parse_args()


q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}
print("Quantization config:", q_config)



def build_model_and_enc(model_path, dtype):
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Note: To avoid OOM after huggingface transformers 4.36.2
    config.use_cache = False
    if "mpt" in config.__class__.__name__.lower():
        enc = AutoTokenizer.from_pretrained(
            config.tokenizer_name, trust_remote_code=True
        )
    else:
        enc = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )

    if args.load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch_dtype, trust_remote_code=True
            )
        real_quantize_model_weight(
            model, w_bit=args.w_bit, q_config=q_config, init_only=True
        )

        model.tie_weights()

        # Infer device map
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )
        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)

        model.eval()
    else:  # fp16 to quantized
        args.run_awq &= not args.load_awq  # if load_awq, no need to run awq
        # Init model on CPU:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, trust_remote_code=True, 
        )

        model.eval()

        if args.run_awq:
            assert args.dump_awq, "Please save the awq results with --dump_awq"

            awq_results = run_awq(
                model,
                enc,
                w_bit=args.w_bit,
                q_config=q_config,
                n_samples=128,
                seqlen=512,
            )
            if args.dump_awq:
                dirpath = os.path.dirname(args.dump_awq)
                os.makedirs(dirpath, exist_ok=True)

                torch.save(awq_results, args.dump_awq)
                print("AWQ results saved at", args.dump_awq)

            exit(0)

        if args.load_awq:
            print("Loading pre-computed AWQ results from", args.load_awq)
            awq_results = torch.load(args.load_awq, map_location="cpu")
            apply_awq(model, awq_results)

        # weight quantization
        if args.w_bit is not None:
            if args.q_backend == "fake":
                assert (
                    args.dump_quant is None
                ), "Need to use real quantization to dump quantized weights"
                pseudo_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
                if args.dump_fake:
                    model.save_pretrained(args.dump_fake)
                    print("Pseudo-quantized models saved at", args.dump_fake)
            elif args.q_backend == "real":  # real quantization
                real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
                if args.dump_quant:
                    if not args.dump_quant.endswith("v2.pt"):
                        print("[Info] Auto-change the dump_quant file name to *v2.pt")
                        args.dump_quant = args.dump_quant.replace(".pt", "-v2.pt")
                    dirpath = os.path.dirname(args.dump_quant)
                    os.makedirs(dirpath, exist_ok=True)

                    print(f"Saving the quantized model at {args.dump_quant}...")
                    torch.save(model.cpu().state_dict(), args.dump_quant)
                    exit(0)
            else:
                raise NotImplementedError

        # Move the model to GPU (as much as possible) for LM evaluation

        device_map = infer_auto_device_map(
            model,
            # TODO: can we remove this?
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
        )
        model = dispatch_model(model, device_map=device_map)

    return model, enc


# The following function is used in a separate process to run the generation test.
def _individual_test(in_data, conn, tokens_to_gen):
    '''Test method content, performed on a separate process.'''
    # Change any content within TEST BEGIN and TEST END to change the testing behavior!
    # TEST BEGIN
    conn.send('IDLE_START')
    print('Running idle period for power baseline')
    sleep(15)
    conn.send('IDLE_END')

    sleep(3) # buffer time

    conn.send('MODEL_LOAD_START')
    model, enc = build_model_and_enc(args.model_path, args.dtype)
    conn.send('MODEL_LOAD_END')

    sleep(3) # buffer time

    conn.send('GENERATE_START')
    output, new_tokens = hf_models.generate_from_input(model, enc, in_data, max_new_tokens=tokens_to_gen)
    conn.send('GENERATE_END')

    # TEST END
    conn.send(f'TOKENS:{len(new_tokens)}')
    print(output)
    conn.close()

def run_input(iterations, num_tokens_to_gen):
    # load input data (text) from the given input file
    print("Loading input text...", end='')
    input_data = ""
    with open(args.inputfile, 'r') as input_file:
        input_data = '\n'.join(input_file.readlines())
    print(f'Got {len(input_data)} characters')
    
    for i in range(iterations):
        # set up datestring for subfolder
        date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        print(f'\n### Beginning ({i+1}/{iterations})')
        test_log = Log()
        test_log.begin(interval=0.1)
        sleep(3) # buffer time
    
        # here we put all of the model loading and usage in a separate process
        # this allows us to cleanly release all memory, both CPU and GPU
        # additionally, a pipe is used to send back timestamped messages for the log
        msg_recv, msg_send = Pipe()
        proc = Process(target=_individual_test, args=[input_data, msg_send, num_tokens_to_gen])
        proc.start()
        while proc.is_alive():
            if msg_recv.poll():
                message = str(msg_recv.recv())
                test_log.add_timestamp(message)
                msg_var = message.split(':')
                if len(msg_var) > 1:
                    if msg_var[0] == 'TOKENS':
                        test_log.tokens_generated = int(msg_var[1])
        proc.join()
        if not msg_send.closed:
            msg_send.close()
        msg_recv.close()
    
        sleep(3) # buffer time
        test_log.end()
        print(f'### Finished ({i+1}/{iterations}), generated {test_log.tokens_generated} tokens')
    
        # save the log to a file for analysis
        outfolder = os.path.join(os.path.abspath(args.outputdir), date_str)
        Path(outfolder).mkdir(parents=True, exist_ok=True)
        log_name_parts = ['log', str(i+1)]
        outfilename = '_'.join(log_name_parts) + '.json'
        outfilepath = os.path.join(outfolder, outfilename)
        print(f'### Saving log to {outfilepath}')
        json_str = test_log.to_json()
        with open(outfilepath, 'w') as fp:
            fp.write(json_str)

def run_tasks():
    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')    
    test_log = Log()
    test_log.begin(interval=0.1)
    print('Running idle period for power baseline')
    sleep(15)
    
    print('MODEL_LOAD_START')
    model, enc = build_model_and_enc(args.model_path, args.dtype)
    print('MODEL_LOAD_END')
    sleep(3) # buffer time
    if args.tasks is not None:
        task_names = args.tasks.split(",")
        lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
        results = evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=task_names,
            batch_size=args.batch_size,
            no_cache=True,
            num_fewshot=args.num_fewshot,
        )
        print(evaluator.make_table(results)) 
        if args.outputdir is not None:
            os.makedirs(os.path.dirname(args.outputdir), exist_ok=True)
            # otherwise cannot save
            results["config"]["model"] = args.model_path
            with open(args.outputdir, "w") as f:
                json.dump(results, f, indent=2)
    test_log.end()

def main():
    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()    
    
    if args.auto_parallel:
        gpu_list = auto_parallel(args)
    
    # get quantization config (apart from w_bit)
    q_config = {
        "zero_point": not args.no_zero_point,  # by default True
        "q_group_size": args.q_group_size,  # whether to use group quantization
    }
    print("Quantization config:", q_config)
    # Dispatch to the correct function
    if args.run_tasks:
        run_tasks()
    elif args.run_input:
        run_input(args.iterations, args.tokens)


if __name__ == "__main__":
    main()
