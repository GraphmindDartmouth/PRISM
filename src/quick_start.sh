#!/usr/bin/env bash

set -euo pipefail

python keyword_generator.py --openai_api_key {api key}

python get_gpt_data.py  --openai_api_key {api key} --data_split train
python get_gpt_data.py  --openai_api_key {api key} --data_split val
python get_gpt_data.py  --openai_api_key {api key} --data_split test

python gpt_rewrite.py --openai_api_key {api key}

python train_model.py

python run_gen.py