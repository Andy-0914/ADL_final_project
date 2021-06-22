import os
from argparse import ArgumentParser
from pathlib import Path
import json
import sys
import copy
from tqdm import tqdm
import logging
import pickle
import random

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_all_context(dialogue):
	all_context = ""
	for turn in dialogue['turns']:
		utternace = turn['utterance']
		all_context += utternace
	return all_context

def main(args):
	preprocess_data_path = "./preprocessed_data/"
	os.makedirs(preprocess_data_path, exist_ok=True)
	preprocessed_data = []

	data_path = args.data_path
	for i in tqdm(range(args.num_data)):
		dialogue_path = args.data_path + "dialogues_" + str(i+1).zfill(3) + ".json"
		with open(dialogue_path) as json_file:
			dialogues = json.load(json_file) #A dialogue file is now loaded

		for dialogue in dialogues: #get dialogue one by one
			dialogue_info = {}
			id_ = dialogue['dialogue_id']
			dialogue_info['id'] = id_
			context = ""
			belief={}
			turns_info = []
			all_context = get_all_context(dialogue)

			for turn in dialogue['turns']:
				speaker = turn['speaker']
				utternace = turn['utterance']
				context += ("<|" + speaker + "|>" + utternace + "<|" + speaker + "|>")

				#track state only if USER is speaking
				if speaker == 'USER':
					try:
						service = turn['frames'][0]['service']
					except:
						#End of dialogue
						break

					slot_values = turn['frames'][0]['state']['slot_values']
					for name, word_list in slot_values.items():
						#Case with multiple reference
						if len(word_list) > 1:
							for word in word_list:
								if word in all_context or word in all_context.lower(): #add this to belief
									belief[service+"-"+name] = word
									break
								if word == word_list[-1]:
									raise ValueError("no word in word_list belongs to context")
						#Case without multiple reference
						else:
							belief[service+"-"+name] = word_list[0]

					#After updating belief state, if belief dict is not empty, we can store the value
					if belief and not args.last_context_only: #choose whether or not to reduce data size
						if random.random() > args.reduce_threshold:
							d = copy.deepcopy(belief)
							turns_info.append({'context': context, 'belief': d, 'service': dialogue['services']})

					elif belief and args.last_context_only:
						if turn == dialogue['turns'][-2]:
							d = copy.deepcopy(belief)
							turns_info.append({'context': context, 'belief': d, 'service': dialogue['services']})

			#add the entire dialogue's turn information into dialogue_info
			dialogue_info['turns'] = turns_info
			#print(dialogue_info)
			#sys.exit()

			if turns_info:
				preprocessed_data.append(dialogue_info)
				#print(dialogue_info)

	print('len: ', len(preprocessed_data))
	with open(preprocess_data_path+"{}.pkl".format(args.mode), 'wb') as f:
		pickle.dump(preprocessed_data, f)
	logging.info(args.mode + ".pkl dumped in " + preprocess_data_path + " directory.")



def parse_args():
	parser = ArgumentParser()
	parser.add_argument(
		"--data_path",
		type=str,
		help="path to directory that stores data to be preprocessed",
		default="../adl-final-dst-with-chit-chat-seen-domains/data-0614/data-0614/dev/",
	)

	parser.add_argument(
		"--num_data",
		type=int,
		help="number of json file to be preprocessed",
		default=20,
		#train=138, dev=20
	)

	parser.add_argument(
		"--mode",
		type=str,
		help="train, dev or test",
		default="dev",
	)

	parser.add_argument(
		"--reduce_threshold",
		type=float,
		help="0 to 1, with 0 mean no reduce",
		default=0.5
	)

	parser.add_argument(
		"--last_context_only",
		type=bool,
		default=True,
	)

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	main(args)