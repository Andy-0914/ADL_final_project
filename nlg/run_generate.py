import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import numpy as np
import json
from tqdm import tqdm

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

parser = argparse.ArgumentParser()
parser.add_argument("--no_cuda", action="store_true", help="avoid using CUDA when available")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--model_name_or_path", type=str, default="output", help="path to pre-trained model or shortcut name")
parser.add_argument("--data", type=str, default="../data/adl-final-dst-with-chit-chat-seen-domains/data-0614/data-0614/test_seen", help="path to data")
parser.add_argument("--output", type=str, help="output file")
parser.add_argument("--eos_token_id", type=int, default=None, help="eos token id")
parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
)
parser.add_argument("--k", type=int, default=0)
parser.add_argument("--p", type=float, default=0.9)
parser.add_argument("--do_sample", action="store_true")

args = parser.parse_args()
print(args)

args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
datafolder = args.data

set_seed(args)

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, pad_token_id=tokenizer.eos_token_id)

model.to(args.device)

fns = os.listdir(datafolder)
print(fns)
outputs = {}
outputs['data'] = []

for fn in tqdm(fns):
    with open(os.path.join(datafolder, fn), "r", encoding='utf8') as f:
        data = json.load(f)
    
    for dialogue in tqdm(data):
        context = ''
        
        for turn in dialogue['turns']:
            speaker = turn['speaker']
            utterance = turn['utterance']

            if speaker == 'SYSTEM':
                document = '<|endoftext|> <|context|> ' + context + ' <|endofcontext|> <|response|> ' + utterance + ' <|endofresponse|> <|chitchat|>'
                input_ids = tokenizer(document, return_tensors="pt")
                input_ids = { k: v.to(args.device) for k, v in input_ids.items() }
                gen_output = model.generate(
                    **input_ids, 
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    do_sample=args.do_sample,
                    max_length=1024
                )
                gen_output = tokenizer.decode(gen_output[0], skip_special_tokens=False)
                chitchat = gen_output.split('<|chitchat|>')[1].split('<|endofchitchat|>')[0]

                outputs['data'] += [{
                    'dialogue_id': dialogue['dialogue_id'],
                    'turn_id': turn['turn_id'],
                    'context': context.replace('<|', ' ').replace('|>', ':'),
                    'response': utterance,
                    'chitchat': chitchat.strip()
                }]

            context += '<|' + speaker.lower() + '|> ' + utterance
        
with open(args.output, "w", encoding='utf8') as f:
    json.dump(outputs, f, indent=1)

