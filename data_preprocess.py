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

def get_user_context(dialogue):
	user_context = ""
	for turn in dialogue['turns']:
		if turn['speaker'] == 'USER':
			utternace = turn['utterance']
			user_context += utternace
	return user_context

def get_system_context(dialogue):
	system_context = ""
	for turn in dialogue['turns']:
		if turn['speaker'] == 'SYSTEM':
			utternace = turn['utterance']
			system_context += utternace
	return system_context

def get_schema(args):
	with open(args.schema_data_path) as json_file:
		schema = json.load(json_file)
	#print(schema[0])
	return schema

def get_service(schema, target_service):
	for service in schema:
		if service['service_name'] == target_service:
			#exteract service information
			context = "<|SLOT|>"
			for slot in service['slots']:
				context += target_service + " " + slot['name'] + "<|SLOT|>"
				#+ " description: " + slot['description'] + "<|SLOT|>"
			#context += "<|SERVICE|>"
			return context

def get_tracking_service(schema, used_service):
	context = ""
	for service in used_service:
		#print(type(get_service(schema,service)))
		context += get_service(schema, service)
	return context

def main(args):
	preprocess_data_path = "./preprocessed_data/"
	os.makedirs(preprocess_data_path, exist_ok=True)
	preprocessed_data = []

	data_path = args.data_path
	schema = get_schema(args)

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
			used_service = []
			user_context = get_user_context(dialogue)
			system_context = get_system_context(dialogue)

			for turn in dialogue['turns']:
				speaker = turn['speaker']
				utternace = turn['utterance']
				#print(turn)
				#sys.exit("ds")

				#track state only if USER is speaking
				if speaker == 'USER':
					context += ("<|" + speaker + "|>" + utternace + "<|" + speaker + "|>")
					belief = {}
					for frame in turn['frames']:
						#print('frame: ', frame)
						try:
							service = frame['service']
							if service not in used_service:
								used_service.append(service)
						except:
							#End of dialogue
							break

						slot_values = frame['state']['slot_values']
						for name, word_list in slot_values.items():
							#Case with multiple reference
							if len(word_list) > 1:
								for word in word_list:
									if word in user_context: #add this to belief
										belief[service+" "+name] = word
										break
									elif word in user_context.lower():
										belief[service+" "+name] = word
										break
									elif word in system_context:
										belief[service+" "+name] = word
										break
									elif word in system_context.lower():
										belief[service+" "+name] = word
										break

									if word == word_list[-1]:
										raise ValueError("no word in word_list belongs to context")
							#Case without multiple reference
							else:
								belief[service+" "+name] = word_list[0]


				#speaker is system
				else:
					response = utternace
					action = ''
					for frame in turn['frames']:
						service = frame['service']
						for act in frame['actions']:
							action += service + ' ' + act['act'].lower() + ' ' + act['slot'] + ' , '

					action = action[:-2]
					if not args.last_context_only: #choose whether or not to reduce data size
						if random.random() > args.reduce_threshold:
							d = copy.deepcopy(belief)
							#print('system belief: ', belief)
							belief_service = [name[:name.find(' ')] for name, value in belief.items()]
							belief_service = dialogue['services']
							#print('belief service: ', belief_service)
							#sys.exit()
							ref_service = get_tracking_service(schema, belief_service)
							#print('ref_service: ', ref_service)
							turns_info.append({'context':'<|CONTEXT|>'+context+'<|ENDOFCONTEXT|>',
												'belief': belief,
												'service': ref_service,
												'response': '<|RESPONSE|>'+response+'<|ENDOFRESPONSE|>'})
							#tracking_service = get_tracking_service(schema, used_service)
							#turns_info.append({'context': context, 'belief': d, 'service': tracking_service})

					elif args.last_context_only:
						if turn == dialogue['turns'][-1]:
							d = copy.deepcopy(belief)
							belief_service = dialogue['services']
							ref_service = get_tracking_service(schema, belief_service)
							turns_info.append({'context':'<|CONTEXT|>'+context+'<|ENDOFCONTEXT|>',
												'belief': belief,
												'service': ref_service,
												'response': '<|RESPONSE|>'+response+'<|ENDOFRESPONSE|>'})

					context += ("<|" + speaker + "|>" + utternace + "<|" + speaker + "|>")




			#add the entire dialogue's turn information into dialogue_info
			dialogue_info['turns'] = turns_info
			#print(dialogue_info)
			#sys.exit()

			if turns_info:
				preprocessed_data.append(dialogue_info)
				#print(dialogue_info)

	#print('data: ', preprocessed_data[0])
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
		default="../adl-final-dst-with-chit-chat-seen-domains/data-0625/data-0625/dev/",
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
		default=0.6
	)

	parser.add_argument(
		"--last_context_only",
		type=bool,
		default=True,
	)

	parser.add_argument(
		"--schema_data_path",
		type=str,
		default="../adl-final-dst-with-chit-chat-seen-domains/data/data/schema.json"
	)

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	main(args)
