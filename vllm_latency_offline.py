#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some parts of this file is adapted from
# https://github.com/vllm-project/vllm/blob/v0.2.1.post1/examples/offline_inference.py
# which is licensed under Apache License 2.0
#
# Copyright 2023 The vLLM team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os
import csv
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--max-new-tokens", type=int, default=1024)
parser.add_argument("--max-input-length", type=int, default=1024)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--tensor-parallel", type=int, default=1)
parser.add_argument('--n',
                    type=int,
                    default=1,
                    help='Number of generated sequences per prompt.')
parser.add_argument('--use-beam-search', action='store_true')
args = parser.parse_args()

from vllm import SamplingParams, LLM
from vllm.engine.arg_utils import EngineArgs

def generate_input(args):
    random_words = ["France" for _ in range(args.max_input_length)]

    input_id = ""

    for word in random_words:
        input_id = input_id + word + " "

    input_id = input_id[:-1]

    input_list = []

    for batch_size in range(args.batch_size):
        input_list.append(input_id)

    return input_list

prompts = generate_input(args)

# Create a sampling params object.
sampling_params = SamplingParams(
    n=args.n,
    temperature=0.0 if args.use_beam_search else 1.0,
    top_p=1.0,
    ignore_eos=True,
    max_tokens=args.max_new_tokens,
)
print(sampling_params)
# Create an LLM.
llm = LLM(model=args.model,
          device="cpu",
          dtype="bfloat16",
          enforce_eager=True,
          tensor_parallel_size=args.tensor_parallel,
          gpu_memory_utilization=0.75,
          trust_remote_code=True,
          distributed_executor_backend="mp")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
start_time = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
end_time = time.perf_counter()
latency = end_time - start_time
total_num_tokens = args.batch_size*(args.max_input_length + args.max_new_tokens)
print("Total Number of Tokens = ", total_num_tokens)

throughput = total_num_tokens/latency
print("Throughput = ", throughput)

list_1 = ["Model Name",
          "precision",
          "throughput",
          "latency",
          "batch size",
          "tensor_parallel",
          "input length",
          "output length"
          ]

list_2 = [args.model,
          'bfloat16',
          throughput,
          latency,
          args.batch_size,
          args.tensor_parallel,
          args.max_input_length,
          args.max_new_tokens,
          ]

assert len(list_1) == len(list_2)

csv_file = "all_models_results.csv"
file_exists = os.path.exists(csv_file)

with open(csv_file, 'a', newline = '') as csvfile:
    writer = csv.writer(csvfile)

    if not file_exists:
        writer.writerow(list_1)

    writer.writerow(list_2)

    csvfile.close()

