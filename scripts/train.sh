K_CENTER=2
K_REFINE=3
K_SKIP=3
MASK_MODE=res #'cat'

L1_LOSS=2
CONTENT_LOSS=2.5e-1
STYLE_LOSS=2.5e-1
PRIMARY_LOSS=0.01
IOU_LOSS=0.25 

INPUT_SIZE=768
DATASET=CLWD
NAME=slbr_v1
# nohup python -u   main.py \
python -u train.py \
 --epochs 100 \
 --schedule 65 \
 --lr 1e-3 \
 --gpu_id 0 \
 --checkpoint 2021/SLBR-Visible-Watermark-Removal/checkpoint \
 --dataset_dir custom_dataset2 \
 --nets slbr  \
 --sltype vggx \
 --mask_mode ${MASK_MODE} \
 --lambda_content ${CONTENT_LOSS} \
 --lambda_style ${STYLE_LOSS} \
 --lambda_iou ${IOU_LOSS} \
 --lambda_l1 ${L1_LOSS} \
 --lambda_primary ${PRIMARY_LOSS} \
 --masked True \
 --loss-type hybrid \
 --models slbr \
 --input-size ${INPUT_SIZE} \
 --crop_size ${INPUT_SIZE} \
 --train-batch 1 \
 --test-batch 1 \
 --preprocess resize \
 --name ${NAME} \
 --k_center ${K_CENTER} \
 --dataset ${DATASET} \
 --use_refine \
 --k_refine ${K_REFINE} \
 --k_skip_stage ${K_SKIP} \
 --start-epoch 71 \
 --resume 2021/SLBR-Visible-Watermark-Removal/checkpoint/slbr_v1/checkpoint.pth.tar \
 --freq 3000
