import pickle
from transformers import(
	GPT2LMHeadModel,
	GPT2TokenizerFast,
	GPT2Tokenizer,
	GPT2Config,
	Trainer,
	TrainingArguments,
)
from dataset import DstDataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import logging
import torch
import sys

logging.basicConfig(
	format="%(asctime)s | %(levelname)s | %(message)s",
	level=logging.INFO,
	datefmt="%Y-%m-%d %H:%M:%S",
)


SPLITS = ['train', 'dev']

class DstTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

		outputs = model(input_ids=inputs['input_ids'].to(device), 
						attention_mask=inputs['attention_mask'].to(device),
						labels=inputs['input_ids'].to(device))
		return outputs.loss


def main():

	#tokenizer = GPT2Tokenizer.from_pretrained("./model/distilgpt2_slot_response/checkpoint-17763")
	#model = GPT2LMHeadModel.from_pretrained("./model/distilgpt2_slot_response/checkpoint-17763")

	tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
	model = GPT2LMHeadModel.from_pretrained("distilgpt2")
	tokenizer.padding_side = "right" # Very Important

	special_tokens_dict = {'pad_token': '<PAD>', 
							'additional_special_tokens': ["<|USER|>", "<|SYSTEM|>", "<|SEP|>", "<|ENDOFTEXT|>", 
							"<|CONTEXT|>", "<|ENDOFCONTEXT|>", "<|BELIEF|>", "<|ENDOFBELIEF|>","<|RESPONSE|>", 
							"<|ENDOFRESPONSE|>", "<|SERVICE|>", "<|ENDOFSERVICE|>", "<|SLOT|>"]}
	num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
	#print('We have added', num_added_toks, 'tokens')
	model.resize_token_embeddings(len(tokenizer))

	dialogues = {}
	for split in SPLITS:
		with open("./preprocessed_data/{}.pkl".format(split), 'rb') as f:
			print('split: ', split)
			dialogues[split] = pickle.load(f)

	data = []
	for split in SPLITS:
		for dialogue in dialogues[split]:
			#print(dialogue)
			for turn in dialogue['turns']:
				#print(turn)
				data.append(turn)
			
	train_dataset = DstDataset(data, tokenizer)
	#eval_dataset = DstDataset(data['dev'], tokenizer)

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)

	training_args = TrainingArguments(
		output_dir=args.output_dir,
		num_train_epochs=2,
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=args.batch_size,
		warmup_steps=500,
		weight_decay=0.01,
		logging_dir=args.logging_dir,
		logging_steps=10,
		learning_rate=1e-4,
		save_strategy=args.save_strategy,
	)

	trainer = DstTrainer(
		model=model,
		tokenizer=tokenizer,
		args=training_args,
		data_collator=train_dataset.collate_fn, 
		train_dataset=train_dataset,
	)

	logging.info('start training')

	trainer.train()


def parse_args():
	parser = ArgumentParser()
	parser.add_argument(
		"--batch_size",
		type=int,
		default=4
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="./model/distilgpt2_slot_response"
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


	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	main()