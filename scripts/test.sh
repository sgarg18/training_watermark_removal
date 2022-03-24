K_CENTER=2
K_REFINE=3
K_SKIP=3
MASK_MODE=res

INPUT_SIZE=512
DATASET=CLWD
NAME=slbr_v1

python3  test.py \
  --nets slbr \
  --models slbr \
  --input-size ${INPUT_SIZE} \
  --crop_size ${INPUT_SIZE} \
  --test-batch 2 \
  --evaluate\
  --dataset_dir custom_dataset \
  --preprocess resize \
  --no_flip \
  --name ${NAME} \
  --mask_mode ${MASK_MODE} \
  --k_center ${K_CENTER} \
  --dataset ${DATASET} \
  --resume 2021/SLBR-Visible-Watermark-Removal/checkpoint/slbr_v1/checkpoint.pth.tar \
  --use_refine \
  --k_refine ${K_REFINE} \
  --k_skip_stage ${K_SKIP} \
  --checkpoint 2021/SLBR-Visible-Watermark-Removal/checkpoint 
  
