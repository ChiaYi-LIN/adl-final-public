CUDA_VISIBLE_DEVICES=2,3 python run_intent_detect.py \
  --model_name_or_path danlou/roberta-large-finetuned-csqa \
  --output_dir ../tmp/intent_detect_roberta_withBot_win3 \
  --overwrite_output_dir True \
  --train_file ../data/train/intent_detect_preprocess_with_win3_and_bot.jsonl \
  --validation_file ../data/validation/intent_detect_preprocess_with_win3_and_bot.jsonl \
  --test_file ../data/validation/intent_detect_preprocess_with_win3_and_bot.jsonl \
  --per_device_train_batch_size 8 \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 3 \
  --logging_steps 500 \
  --max_seq_length 128 \
  --learning_rate 2e-5 \
  --evaluation_strategy steps \
  --save_total_limit 1 \
  