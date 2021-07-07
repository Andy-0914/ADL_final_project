import pickle
from transformers import(
	GPT2LMHeadModel,
	GPT2TokenizerFast,
	GPT2Tokenizer,
	GPT2Config,
	Trainer,
	TrainingArguments,
)
from eval_dataset import DstEvalDataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import logging
import torch
import sys
from tqdm import tqdm
import json

logging.basicConfig(
	format="%(asctime)s | %(levelname)s | %(message)s",
	level=logging.INFO,
	datefmt="%Y-%m-%d %H:%M:%S",
)


SPLITS = ['train', 'dev']


def main():

	tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
	model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)

	"""tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
	model = GPT2LMHeadModel.from_pretrained("distilgpt2")
	tokenizer.padding_side = "right" # Very Important

	special_tokens_dict = {'pad_token': '<PAD>', 
							'additional_special_tokens': ["<|USER|>", "<|SYSTEM|>", "<|SEP|>", "<|ENDOFTEXT|>", 
							"<|CONTEXT|>", "<|ENDOFCONTEXT|>", "<|BELIEF|>", "<|ENDOFBELIEF|>","<|RESPONSE|>", 
							"<|ENDOFRESPONSE|>", "<|SERVICE|>", "<|ENDOFSERVICE|>", "<|SLOT|>"]}
	num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
	#print('We have added', num_added_toks, 'tokens')
	model.resize_token_embeddings(len(tokenizer))"""


	dialogues = {}
	with open("./preprocessed_data/{}.pkl".format(args.mode), 'rb') as f:
		dialogues['dev'] = pickle.load(f)
		#print(dialogues['dev'][0])
		#sys.exit()

	data = []
	for dialogue in dialogues['dev']:
		#print(dialogue)
		data.append(dialogue)
		#print(data[0].items())
		#sys.exit()
		#for turn in dialogue['turns']:
		#	data.append(turn)
			
	#train_dataset = DstDataset(data['train'], tokenizer)
	eval_dataset = DstEvalDataset(data, tokenizer)

	evalloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn)

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)
	model.eval()


	ans = {}
	accuracy_loss = 0
	#count = 0

	for num, batch_input in enumerate(tqdm(evalloader)):
		#print('input: ', tokenizer.decode(batch_input['input_ids'][0]))
		outputs = model.generate(batch_input['input_ids'].to(device), num_beams=1, max_length=1024,
								 attention_mask=batch_input['attention_mask'].to(device))
		#outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
		outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
		outputs = outputs[0]
		#print('outputs: ', outputs)
		outputs = outputs[outputs.find("<|SEP|>"): outputs.find("<|ENDOFTEXT|>")]
		#print('after_outputs: ', outputs)
		outputs_dict = {}
		for text in outputs.split("<|SEP|>"):
			if "=" in text:
				text = text.strip()
				outputs_dict[text[:text.find("=")]] = text[text.find("=")+1:]

		#print('eval_dataset: ', eval_dataset[num]['turns'][0])
		#print('outputs_dict: ', outputs_dict)
		#continue
		#print('eval: ', eval_dataset[num]['turns'][0])

		if eval_dataset[num]['turns'][0]['belief'] != outputs_dict:
			print('ground_truth: ', eval_dataset[num]['turns'][0]['belief'])
			print('outputs_dict: ', outputs_dict)
			accuracy_loss += 1

		outputs_dict = {}
		for text in outputs.split("<|SEP|>"):
			if "=" in text:
				text = text.strip()
				outputs_dict[text[:text.find(" ")] + "-" + text[text.find(" ")+1:text.find("=")]] = text[text.find("=")+1:]

		ans[eval_dataset[num]['id']] = outputs_dict
		print('eval_loss: ', accuracy_loss)

		#print('after_output_dict: ', outputs_dict)

		#count += 1
		#if count == 3:
		#	break

	print(accuracy_loss / len(evalloader))


	with open("submission.json", 'w', encoding='utf-8') as json_file:
		json.dump(ans, json_file)


def parse_args():
	parser = ArgumentParser()
	parser.add_argument(
		"--batch_size",
		type=int,
		default=1
	)

	parser.add_argument(
		"--model_checkpoint",
		type=str,
		default="./model/distilgpt2_slot_response/checkpoint-35524"
		#checkpoint-14736 loss=0.8499
		#checkpoint-44208 loss=0.7882
	)

	parser.add_argument(
		"--mode",
		type=str,
		default="dev"
	)


	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	main()