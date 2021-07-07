import torch
import sys

class DstEvalDataset(torch.utils.data.Dataset):
	def __init__(self, data, tokenizer):
		self.data = data
		self.tokenizer = tokenizer


	def __getitem__(self, idx):
		return self.data[idx]


	def __len__(self):
		return len(self.data)


	def collate_fn(self, samples):
		encoder_inputs = []
		#print(samples)
		samples = samples[0]['turns']
		#print(samples)
		for sample in samples:
			#print('sample: ', sample)
			#sys.exit()
			context = sample['context'] + "<|SERVICE|>" + sample['service'] + "<|ENDOFSERVICE|>" + "<|BELIEF|>" + "<|SEP|>"

			encoder_inputs.append(context)
		#print('encoder_inputs: ', encoder_inputs)
		#sys.exit()



		encoder_inputs = self.tokenizer(encoder_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024)


		return encoder_inputs


