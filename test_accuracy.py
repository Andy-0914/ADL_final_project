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
		outputs = model.generate(batch_input['input_ids'].to(device), num_beams=1, max_length=1024,
								 attention_mask=batch_input['attention_mask'].to(device))
		outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
		outputs = outputs[0]
		#print('outputs: ', outputs)
		outputs = outputs[outputs.find("<|SEP|>"):]

		modified_output = "<|SEP|>"
		for text in outputs.split("<|SEP|>"):
			if "=" in text:
				modified_output += text + "<|SEP|>"
		#print('modified_output: ', modified_output)

		ground_truth = eval_dataset[num]['belief'].items()
		ground_truth_context = "<|SEP|>"
		for state, value in ground_truth:
			ground_truth_context += state + "=" + value + "<|SEP|>"


		if ground_truth_context != modified_output:
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
		default="./model/distilgpt2/checkpoint-8566"
	)


	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	main()