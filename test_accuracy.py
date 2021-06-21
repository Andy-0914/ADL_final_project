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

logging.basicConfig(
	format="%(asctime)s | %(levelname)s | %(message)s",
	level=logging.INFO,
	datefmt="%Y-%m-%d %H:%M:%S",
)


SPLITS = ['train', 'dev']


def main():

	tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
	model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)

	dialogues = {}
	with open("./preprocessed_data/dev.pkl", 'rb') as f:
		dialogues['dev'] = pickle.load(f)

	data = []
	for dialogue in dialogues['dev']:
		for turn in dialogue['turns']:
			data.append(turn)
			
	#train_dataset = DstDataset(data['train'], tokenizer)
	eval_dataset = DstEvalDataset(data, tokenizer)

	evalloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn)

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)
	model.eval()

	accuracy_loss = 0
	for num, batch_input in enumerate(tqdm(evalloader)):
		#print('input: ', tokenizer.decode(batch_input['input_ids'][0]))
		outputs = model.generate(batch_input['input_ids'].to(device), num_beams=1, max_length=1024,
								 attention_mask=batch_input['attention_mask'].to(device))
		#outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
		outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
		outputs = outputs[0]
		#print('outputs: ', outputs)
		outputs = outputs[outputs.find("<|SEP|>"): outputs.find("<|ENDOFTEXT|>")]
		#print('after_outputs: ', modified_output)
		outputs_dict = {}
		for text in outputs.split("<|SEP|>"):
			if "=" in text:
				text = text.strip()
				outputs_dict[text[:text.find("=")]] = text[text.find("=")+1:]

		if eval_dataset[num]['belief'] != outputs_dict:
			print('ground_truth: ', eval_dataset[num]['belief'])
			print('outputs_dict: ', outputs_dict)
			accuracy_loss += 1

		print('eval_loss: ', accuracy_loss)

	print(accuracy_loss / len(evalloader))



def parse_args():
	parser = ArgumentParser()
	parser.add_argument(
		"--batch_size",
		type=int,
		default=1
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="./model/distilgpt2"
	)
	parser.add_argument(
		"--logging_dir",
		type=str,
		default="./logging"
	)
	parser.add_argument(
		"--save_strategy",
		type=str,
		default="epoch"
	)
	parser.add_argument(
		"--model_checkpoint",
		type=str,
		default="./model/distilgpt2/checkpoint-14736"
	)


	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	main()