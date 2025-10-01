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
from datasets import load_dataset
from torch import nn
import tqdm
import os
import sys

# set up default variables
models_filepath = 'models.txt'
input_filepath = 'input.txt'
iterations = 5
output_dir = 'out'
tags = list()
opt_no_quant = False
num_tokens_to_gen = 64
is_dry = False

# function for printing the usage text
parser = argparse.ArgumentParser(
    description="Run LLM tests with configurable options."
)

parser.add_argument("--model_path", type=str,
                    help="Path of the hf model")

parser.add_argument("--inputfile", type=str, default="./input.txt",
                    help="File used as input for text generation (Default: ./input.txt)")

parser.add_argument("--iterations", type=int, default=5,
                    help="Number of iterations to repeat individual tests (Default: 5)")

parser.add_argument("--outputdir", type=str, default="./out",
                    help="Directory to output log files (Default: ./out)")

parser.add_argument("--tag", action="append", default=[],
                    help="Add a tag to the generated log files, can be called multiple times")

parser.add_argument("--tokens", type=int, default=64,
                    help="Number of tokens to generate (Default: 64)")

# New options (quantization / training)
parser.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16"])

parser.add_argument("--batch_size", type=int, default=1,
                    help="Batch size")

parser.add_argument("--tasks", type=str, default=None,
                    help="Tasks to evaluate")

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

# VILA-specific options
parser.add_argument("--vila-15", action="store_true",
                    help="Quantizing VILA 1.5")

parser.add_argument("--vila-20", action="store_true",
                    help="Quantizing or smoothing VILA 2.0 (NVILA)")

parser.add_argument("--smooth_scale", action="store_true",
                    help="Generate the act scale of visiontower")

parser.add_argument("--media_path", type=str, nargs="+",
                    help="Input video(s) to get act scale for visiontower")

parser.add_argument("--act_scale_path", type=str, default=None,
                    help="Path to save act scale")


def build_model_and_enc(model_path, dtype):
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    if vila_10_quant_mode:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        enc, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device="cpu",
            **{"use_cache": False},
        )
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
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
        kwargs = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True}
        if not vila_10_quant_mode:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, config=config, trust_remote_code=True, **kwargs
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
        kwargs = {
            "max_memory": get_balanced_memory(
                model, max_memory if len(max_memory) > 0 else None
            )
        }
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
            **kwargs,
        )
        model = dispatch_model(model, device_map=device_map)

    return model, enc


# load models from the list of names in the given file
print("Loading model names...", end='')
models = []
with open(models_filepath, 'r') as model_file:
    for m in model_file.readlines():
        models.append(m.rstrip('\n'))
print(f'Got {len(models)} models')

# load input data (text) from the given input file
print("Loading input text...", end='')
input_data = ""
with open(input_filepath, 'r') as input_file:
    input_data = '\n'.join(input_file.readlines())
print(f'Got {len(input_data)} characters')

# set up suffix from tags
suffix = '_'.join(tags)

# dryness check
if is_dry:
    print(f'Output directory: {os.path.abspath(output_dir)}')
    print(f'# of tokens to generate: {num_tokens_to_gen}')
    print(f'# of iterations: {iterations}')
    print(f'4-bit quantize? {"NO" if opt_no_quant else "YES"}')
    print(f'Models file: {os.path.abspath(models_filepath)}')
    print(f'Input file: {os.path.abspath(input_filepath)}')
    print(f'Suffix: {suffix}')
    exit(0)


# post-argument-checking imports (to prevent time delay)
import hf_models
from statlog import Log

from datetime import datetime
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path
from time import sleep

# set up datestring for subfolder
date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')

# login to huggingface hub (for access to gated models)
hf_models.login_by_token()

# ensure models are loaded in the cache
# we do not want to benchmark network download times!
for m in models:
    hf_models.download_model(m)
print("Download(s) complete")
sleep(3)


# The following function is used in a separate process to run the generation test.
# Add/change any desired testing functionality in this function to ensure it is tested on each model!
def _individual_test(model_name: str, in_data, conn: Connection, do_quantize: bool, tokens_to_gen: int):
    '''Test method content, performed on a separate process.'''
    # Change any content within TEST BEGIN and TEST END to change the testing behavior!
    # TEST BEGIN

    conn.send('IDLE_START')
    print('Running idle period for power baseline')
    sleep(15)
    conn.send('IDLE_END')

    sleep(3) # buffer time

    conn.send('MODEL_LOAD_START')
    mdl = None
    tk = None
    if do_quantize:
        mdl, tk = hf_models.load_model_quantized(model_name)
    else:
        mdl, tk = hf_models.load_model(model_name)
    conn.send('MODEL_LOAD_END')

    sleep(3) # buffer time

    conn.send('GENERATE_START')
    output, new_tokens = hf_models.generate_from_input(mdl, tk, in_data, max_new_tokens=tokens_to_gen)
    conn.send('GENERATE_END')

    # TEST END
    conn.send(f'TOKENS:{len(new_tokens)}')
    print(output)
    conn.close()


# run tests
for m in models:
    for i in range(iterations):
        m_subname = m.split('/')[-1]
        print(f'\n### Beginning test of {m_subname} ({i+1}/{iterations})')

        test_log = Log()
        test_log.begin(interval=0.1)
        sleep(3) # buffer time

        # here we put all of the model loading and usage in a separate process
        # this allows us to cleanly release all memory, both CPU and GPU
        # additionally, a pipe is used to send back timestamped messages for the log
        msg_recv, msg_send = Pipe()
        proc = Process(target=_individual_test, args=[m, input_data, msg_send, not opt_no_quant, num_tokens_to_gen])
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
        print(f'### Finished test of {m_subname} ({i+1}/{iterations}), generated {test_log.tokens_generated} tokens')

        # save the log to a file for analysis
        outfolder = os.path.join(os.path.abspath(output_dir), date_str)
        Path(outfolder).mkdir(parents=True, exist_ok=True)
        log_name_parts = ['log', m_subname, str(i+1)]
        if opt_no_quant:
            log_name_parts.append('no-quant')
        if len(suffix) > 0:
            log_name_parts.append(suffix)
        outfilename = '_'.join(log_name_parts) + '.json'
        outfilepath = os.path.join(outfolder, outfilename)
        print(f'### Saving log to {outfilepath}')
        json_str = test_log.to_json()
        with open(outfilepath, 'w') as fp:
            fp.write(json_str)
