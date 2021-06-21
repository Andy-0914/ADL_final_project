import torch

class DstDataset(torch.utils.data.Dataset):
	def __init__(self, data, tokenizer):
		self.data = data
		self.tokenizer = tokenizer

		#print('data: ', self.data)
		#enoceder_context = [sample['context'] for sample in self.data]
		#encoder_state = [sample['belief'] for sample in self.data]


	def __getitem__(self, idx):
		return self.data[idx]


	def __len__(self):
		return len(self.data)


	def collate_fn(self, samples):
		#enoceder_context = [sample['context'] for sample in samples]
		#encoder_state = [sample['belief'] for sample in samples]
		#encoder_inputs = [sample['context']+sample['belief'] for sample in samples]
		encoder_inputs = []
		for sample in samples:
			context = sample['context'] + "<|SEP|>"
			#print(type(sample['belief'].items()))
			dict_item = sample['belief'].items()
			for state, value in dict_item:
				context += state + "=" + value + "<|SEP|>"
			#print('context: ', context)
			encoder_inputs.append(context)


		#print('encoder_context: ', enoceder_context)
		#print('encoder_state: ', encoder_state)

		"""encoder_inputs = self.tokenizer(enoceder_context, return_tensors="pt", padding=True, truncation=True, max_length=1024)#padding=True, truncation=True
		with self.tokenizer.as_target_tokenizer():
			labels = self.tokenizer(encoder_state, return_tensors="pt", padding=True, truncation=True,max_length=1024)"""

		#print('encoder_inputs: ', encoder_inputs)
		#print('labels: ', labels)

		"""inputs = {'input_ids': encoder_inputs['input_ids'], 
					'attention_mask': encoder_inputs['attention_mask'],
					'labels': labels}"""

		encoder_inputs = self.tokenizer(encoder_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
		#print('encoder_inputs: ', encoder_inputs)


		return encoder_inputs


