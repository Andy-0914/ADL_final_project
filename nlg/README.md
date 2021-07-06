## preprocessing

```
python preprocess.py --data <data>
```

* **data**: your data path (ex: ../data/adl-final-dst-with-chit-chat-seen-domains/data-0614/data-0614)

## train clm

```
python3 run_language_modeling.py \
  --model_type <model_type> \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --train_file <train_file> \
  --validation_file <validation_file> \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 18 \
  --num_train_epochs 10 \
  --learning_rate 1e-3 \
  --fp16 \
  --prediction_loss_only \
```

* **model_type**: Model type from huggingface.co/models (ex: gpt2)
* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models (ex: gpt2)
* **output_dir**: The output directory where the model predictions and checkpoints will be written. (ex: gpt2_clm)
* **train_file**: The input training data file (a text file). (ex: ./lm.input.train.txt)
* **validation_file**: An optional input evaluation data file to evaluate the perplexity on (a text file). (ex: ./lm.input.dev.txt)

## train accentor

```
python ./run_multiple_choice.py \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --train_file <train_file> \
  --validation_file <validation_file> \
  --do_train \
  --do_eval \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 512 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 12 \
  --per_device_eval_batch_size 2 \
  --eval_accumulation_steps 128 \
  --logging_strategy steps \
  --logging_steps 100 \
  --evaluation_strategy steps \
  --eval_steps 2000 \
  --save_strategy steps \
  --save_steps 2000 \
  --save_total_limit 10 \
  --warmup_ratio 0.1 \
  --fp16 \
```

* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models (ex: roberta-base)
* **output_dir**: The output directory where the model predictions and checkpoints will be written. (ex: arranger_roberta_base)
* **train_file**: The input training data file (a text file). (ex: ./arranger_input.train.json)
* **validation_file**: An optional input evaluation data file to evaluate the perplexity on (a text file). (ex: ./arranger_input.dev.json)


## Chitchat generate

Run generater

```
python run_generate.py \
  --model_name_or_path <model_name_or_path> \
  --data <data> \
  --output <output> \
  [--temperature <temperature>] \
  [--k <k>] \
  [--p <p>] \
  [--num_beams <num_beams>] \
  [--do_sample] \
```

* **model_name_or_path**: Path to model (ex: ./gpt2_clm/)
* **data**: path to data. (ex: ./data/test_seen/)
* **output**: output file. (ex: ./test_seen_chitchat.json)
* **num_beams**: Number of beams to use for decoding. EX: 5
* **k**: The number of highest probability vocabulary tokens to keep for top-k-filtering. EX: 50
* **p**: If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation. EX: 0.9
* **temperature**: The value used to module the next token probabilities. EX: 0.75
* **do_sample**: Whether or not to use sampling ; use greedy decoding otherwise.

Run accentor

```
python ./run_multiple_choice.py \
  --model_name_or_path <model_name_or_path> \
  --output_dir arranger_roberta_base/ \
  --output_file <output_file> \
  --test_file <test_file> \
  --do_predict \
  --max_seq_length 512 \
```

* **model_name_or_path**: Path to model (ex: ./arranger_roberta_base/)
* **output_file**: path to data. (ex: ./nlp_output.json)
* **test_file**: test file for accentor. (ex: ./test_seen_chitchat.json)

## Reproduce our result
```
bash download.sh

python run_generate.py \
  --model_name_or_path gpt2_clm \
  --data <data> \
  --output test_seen_chitchat.json \
  --num_beams 3

python run_multiple_choice.py \
  --model_name_or_path arranger_roberta_base/ \
  --output_dir arranger_roberta_base/ \
  --output_file <output_file> \
  --test_file test_seen_chitchat.json \
  --do_predict \
  --max_seq_length 512 \
  --per_device_eval_batch_size 4 
```

or 

```
bash download.sh
bash run_nlg.sh <data> <output_file>
```

* **data**: path to data. (ex: ./data/test_seen/)
* **output_file**: path to data. (ex: ./nlp_output.json)