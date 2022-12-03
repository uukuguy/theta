train:
	python run.py \
		--do_train \
		--train_file data/train.json \
		--val_file data/val.json

eval:
	python run.py \
		--do_eval \
		--val_file data/val.json

predict:
	python run.py \
		--do_predict \
		--test_file data/val.json
eda:
	PYTHONPATH=$PWD python -m theta.nlp.eda run
