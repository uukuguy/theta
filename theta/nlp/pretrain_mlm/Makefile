SEED=42
BATCH_SIZE=12
MAX_SEQ_LENGTH=512

# #EPOCHS=10
# MAX_TRAIN_STEPS=125000
# SAVE_STEPS=1000
# WARMUP_STEPS=1000

MAX_TRAIN_STEPS=2000
SAVE_STEPS=100
WARMUP_STEPS=200

LR=2e-4
ADAM_EPS=1e-6
WEIGHT_DECAY=0.01
GRAD_ACCU_STEPS=1
LOGGING_STEPS=1000
EVAL_STEPS=1000
GRADIENT_ACCUMULATION_STEPS=16
MLM_PROBABILITY=0.15
RESUME_FROM_CHECKPOINT=outputs/auto_kg_bert_base/step_600

prepare_data:
	python prepare_data.py

bert-base:
	python run_pretrain_mlm.py \
		--model_name_or_path=bert-base-chinese \
		--output_dir outputs/auto_kg_bert_base \
		--train_file ./lm_train.txt \
		--validation_file ./lm_val.txt \
		--max_seq_length ${MAX_SEQ_LENGTH} \
		--per_device_train_batch_size ${BATCH_SIZE} \
		--gradient_accumulation_steps ${GRAD_ACCU_STEPS} \
		--learning_rate ${LR} \
		--max_train_steps ${MAX_TRAIN_STEPS} \
		--num_warmup_steps ${WARMUP_STEPS} \
		--checkpointing_steps ${SAVE_STEPS} \
		--mlm_probability ${MLM_PROBABILITY} \
		--overwrite_cache \
		--seed ${SEED} \
		--weight_decay ${WEIGHT_DECAY} \
		--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
		--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT} \
		--with_tracking

mlm_ddp:
	accelerate launch \
		run_pretrain_mlm.py \
		--model_name_or_path=bert-base-chinese \
		--output_dir outputs/auto_kg_bert_base \
		--train_file ./lm_train.txt \
		--validation_file ./lm_val.txt \
		--max_seq_length ${MAX_SEQ_LENGTH} \
		--per_device_train_batch_size ${BATCH_SIZE} \
		--gradient_accumulation_steps ${GRAD_ACCU_STEPS} \
		--learning_rate ${LR} \
		--max_train_steps ${MAX_TRAIN_STEPS} \
		--num_warmup_steps ${WARMUP_STEPS} \
		--checkpointing_steps ${SAVE_STEPS} \
		--mlm_probability ${MLM_PROBABILITY} \
		--overwrite_cache \
		--seed ${SEED} \
		--weight_decay ${WEIGHT_DECAY} \
		--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
		--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT} \
		--with_tracking

