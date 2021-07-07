# export CUDA_VISIBLE_DEVICES=0

python run_generate.py \
  --model_name_or_path gpt2_clm \
  --data $1 \
  --output test_seen_chitchat.json \
  --num_beams 3

python run_multiple_choice.py \
  --model_name_or_path arranger_roberta_base/ \
  --output_dir arranger_roberta_base/ \
  --output_file $2 \
  --test_file test_seen_chitchat.json \
  --do_predict \
  --max_seq_length 512 \
  --per_device_eval_batch_size 4 