# ADL_final_project

run python data_preprocess.py --data_path path/to/training/data/directory --num_data 138 --mode train --schema_data_path path/to/schema/data to preprocess training data

run python data_preprocess.py --data_path path/to/dev/data/directory --num_data 20 --mode dev --schema_data_path path/to/schema/data to preprocess dev data

run python main.py for training. A model checkpoint should be created in the ./model/distilgpt2_slot_response directory. Say the checkpoint is checkpoint-10000.

run python test_data_preprocess.py --data_path path/to/test_seen/directory --num_data 16 --mode test_seen --schema_data_path path/to/schema/data/path to prepare for test seen data.

run python test_accuracy --model_checkpoint ./model/distilgpt2_slot_response/checkpoint-10000 --mode test_seen to test the seen case.

To test the unseen case, change all test_seen context above to text_unseen.
