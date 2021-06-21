import torch

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
		for sample in samples:
			encoder_inputs.append(sample['context'])


		encoder_inputs = self.tokenizer(encoder_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024)


		return encoder_inputs


