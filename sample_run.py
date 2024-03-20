torchrun --nproc_per_node 2 \
-m FlagEmbedding.baai_general_embedding.retromae_pretrain.run \
--output_dir examples/finetune/model/ \
--model_name_or_path BAAI/bge-large-en \
--train_data ../pretrain_data/pretrain_data.jsonl \
--learning_rate 2e-5 \
--num_train_epochs 2 \
--per_device_train_batch_size 1 \
--dataloader_drop_last True \
--max_seq_length 512 \
--logging_steps 10 \
--dataloader_num_workers 12



# python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
# --model_name_or_path BAAI/bge-large-zh-v1.5 \
# --input_file examples/finetune/toy_finetune_data.jsonl \
# --output_file examples/finetune/toy_finetune_data_minedHN.jsonl \
# --range_for_sampling 2-200 \
# --negative_number 15 \
# --use_gpu_for_searching 


# torchrun --nproc_per_node 2 \
# -m FlagEmbedding.baai_general_embedding.finetune.run \
# --output_dir examples/finetune/model/ \
# --model_name_or_path BAAI/bge-large-zh-v1.5 \
# --train_data examples/finetune/toy_finetune_data.jsonl \
# --learning_rate 1e-5 \
# --fp16 \
# --num_train_epochs 5 \
# --per_device_train_batch_size 1 \
# --dataloader_drop_last True \
# --normlized True \
# --temperature 0.02 \
# --query_max_len 64 \
# --passage_max_len 512 \
# --train_group_size 2 \
# --negatives_cross_device \
# --logging_steps 10 \
# --query_instruction_for_retrieval ""
