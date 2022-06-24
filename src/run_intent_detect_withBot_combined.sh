CUDA_VISIBLE_DEVICES=2,3 python run_intent_detect.py \
  --model_name_or_path bert-base-cased \
  --output_dir ../tmp/intent_detect_bert_withBot_combined \
  --overwrite_output_dir True \
  --train_file ../data/combined/intent_detect_preprocess_withBot.jsonl \
  --validation_file ../data/validation/intent_detect_preprocess_withBot.jsonl \
  --per_device_train_batch_size 8 \
  --do_train \
  --num_train_epochs 3 \
  --logging_steps 500 \
  --max_seq_length 128 \
  --learning_rate 2e-5 \
  --save_total_limit 1 \

# validation_file argument is entered to avoid error only