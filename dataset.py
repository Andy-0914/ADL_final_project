import torch
import sys

class DstDataset(torch.utils.data.Dataset):
	def __init__(self, data, tokenizer):
		self.data = data
		self.tokenizer = tokenizer

	def __getitem__(self, idx):
		return self.data[idx]

	def __len__(self):
		return len(self.data)

	def collate_fn(self, samples):
		encoder_inputs = []
		for sample in samples:
			#print('sample: ', sample)
			context = "<|ENDOFTEXT|>" + sample['context']
			context += "<|SERVICE|>" + sample['service'] + "<|ENDOFSERVICE|>"
			context += "<|BELIEF|><|SEP|>"
			#context += sample['service'] + "<|SEP|>"

			dict_item = sample['belief'].items()
			for state, value in dict_item:
				context += state + "=" + value + "<|SEP|>"
			#print('context: ', context)
			#sys.exit()
			context += "<|ENDOFBELIEF|>" + sample['response']
			#print('context: ', context)
			encoder_inputs.append(context + "<|ENDOFTEXT|>")
			print('context: ', encoder_inputs[0])
			sys.exit()


		encoder_inputs = self.tokenizer(encoder_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
		#print('encoder_inputs: ', encoder_inputs)


		return encoder_inputs


