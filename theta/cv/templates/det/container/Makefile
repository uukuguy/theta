TASK_NAME=underwater_detection
PRETRAINED_MODEL_NAME=cascade_rcnn_fpn_coco
# PRETRAINED_MODEL_NAME=cascade_mask_rcnn_swin_small_3x
# 0.42839
# PRED_THRESHOLD=0.3
# SUBMIT_THRESHOLD=0.3
PRED_THRESHOLD=0.2
SUBMIT_THRESHOLD=0.2

help:
	@echo "make [train|predict|submit]"

prepare_data:
	python data/prepare_data.py

train:
	PYTHONPATH=${PWD}/.. python train.py \
			   configs/${TASK_NAME}_cascade_rcnn_fpn_coco.py \
			   --gpu-ids 0 \
			   --seed 42

predict:
	PYTHONPATH=${PWD}/.. python test.py \
			   work_dirs/${TASK_NAME}_cascade_rcnn_fpn_coco/latest/${TASK_NAME}_cascade_rcnn_fpn_coco.py \
			   work_dirs/${TASK_NAME}_cascade_rcnn_fpn_coco/latest/best_bbox_mAP.pth \
			   --predict \
			   --task_name ${TASK_NAME} \
			   --show-score-thr ${PRED_THRESHOLD} \
			   --submission_file submissions/submission_${TASK_NAME}_cascade_rcnn.json \
			   --out work_dirs/preds_${TASK_NAME}_cascade_rcnn.pkl \

			   # --show-dir work_dirs/${TASK_NAME}_preds
				

submit:
	PYTHONPATH=${PWD}/.. python test.py \
			   work_dirs/${TASK_NAME}_cascade_rcnn_fpn_coco/latest/${TASK_NAME}_cascade_rcnn_fpn_coco.py \
			   work_dirs/${TASK_NAME}_cascade_rcnn_fpn_coco/latest/best_bbox_mAP.pth \
			   --submit \
			   --task_name ${TASK_NAME} \
			   --test_json_file "data/test/annotations/test_coco.json" \
			   --prob_threshold ${SUBMIT_THRESHOLD} \
			   --min_size 2 \
			   --submission_file submissions/submission_${TASK_NAME}_cascade_rcnn.json \
			   --out work_dirs/preds_${TASK_NAME}_cascade_rcnn.pkl \

train_swin:
	PYTHONPATH=${PWD}/.. python train_swin.py \
			   configs/${TASK_NAME}_${PRETRAINED_MODEL_NAME}.py \
			   --cfg-options model.pretrained=../weights/swin/cascade_mask_rcnn_swin_small_patch4_window7.pth \
			   --gpu-ids 0 \
			   --seed 42


# -------------------- DetectoRS --------------------
train_detectors:
	PYTHONPATH=${PWD}/.. python train.py \
			   configs/${TASK_NAME}_detectors_cascade_rcnn_coco.py \
			   --gpu-ids 0 \
			   --seed 42

predict_detectors:
	PYTHONPATH=${PWD}/.. python test.py \
			   work_dirs/${TASK_NAME}_detectors_cascade_rcnn_coco/latest/${TASK_NAME}_detectors_cascade_rcnn_coco.py \
			   work_dirs/${TASK_NAME}_detectors_cascade_rcnn_coco/latest/best_bbox_mAP.pth \
			   --predict \
			   --task_name ${TASK_NAME} \
			   --show-score-thr ${PRED_THRESHOLD} \
			   --submission_file submissions/submission_${TASK_NAME}_detectors.json \
			   --out work_dirs/preds_${TASK_NAME}_detectors.pkl \

			   # --show-dir work_dirs/${TASK_NAME}_preds
			   #
submit_detectors:
	PYTHONPATH=${PWD}/.. python test.py \
			   work_dirs/${TASK_NAME}_detectors_cascade_rcnn_coco/latest/${TASK_NAME}_detectors_cascade_rcnn_coco.py \
			   work_dirs/${TASK_NAME}_detectors_cascade_rcnn_coco/latest/best_bbox_mAP.pth \
			   --submit \
			   --task_name ${TASK_NAME} \
			   --test_json_file "data/test/annotations/test_coco.json" \
			   --prob_threshold ${SUBMIT_THRESHOLD} \
			   --min_size 2 \
			   --submission_file submissions/submission_${TASK_NAME}_detectors.json \
			   --out work_dirs/preds_${TASK_NAME}_detectors.pkl \

# -------------------- ResNeSt --------------------
train_resnest:
	PYTHONPATH=${PWD}/.. python train.py \
			   configs/${TASK_NAME}_resnest_cascade_rcnn_coco.py \
			   --gpu-ids 0 \
			   --seed 42

predict_resnest:
	PYTHONPATH=${PWD}/.. python test.py \
			   work_dirs/${TASK_NAME}_resnest_cascade_rcnn_coco/latest/${TASK_NAME}_resnest_cascade_rcnn_coco.py \
			   work_dirs/${TASK_NAME}_resnest_cascade_rcnn_coco/latest/best_bbox_mAP.pth \
			   --predict \
			   --task_name ${TASK_NAME} \
			   --show-score-thr ${PRED_THRESHOLD} \
			   --submission_file submissions/submission_${TASK_NAME}_resnest.json \
			   --out work_dirs/preds_${TASK_NAME}_resnest.pkl \

			   # --show-dir work_dirs/${TASK_NAME}_preds
			   #
submit_resnest:
	PYTHONPATH=${PWD}/.. python test.py \
			   work_dirs/${TASK_NAME}_resnest_cascade_rcnn_coco/latest/${TASK_NAME}_resnest_cascade_rcnn_coco.py \
			   work_dirs/${TASK_NAME}_resnest_cascade_rcnn_coco/latest/best_bbox_mAP.pth \
			   --submit \
			   --task_name ${TASK_NAME} \
			   --test_json_file "data/test/annotations/test_coco.json" \
			   --prob_threshold ${SUBMIT_THRESHOLD} \
			   --min_size 2 \
			   --submission_file submissions/submission_${TASK_NAME}_resnest.json \
			   --out work_dirs/preds_${TASK_NAME}_resnest.pkl \

# -------------------- PaddleDetection --------------------
train_ppdet:
	python train_ppdet.py \
		-c configs/ppdet/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.yml \
		--eval

predict_ppdet:
	python export_model.py \
		-c configs/ppdet/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.yml \
		-o weight=./output/cascade_rcnn_r50_fpn_1x_coco/best_model \
	python deploy/python/infer.py \
	--model_dir=./inference/cascade_rcnn_r50_fpn_1x_coco \
	--image_dir=./data/test/images \
	--use_gpu=True

build_docker:
	docker build -f Dockerfile -t ootb/xray .

