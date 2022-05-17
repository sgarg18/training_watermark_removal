K_CENTER=2
K_REFINE=3
K_SKIP=3
MASK_MODE=res


INPUT_SIZE=768
NAME=slbr_v1
TEST_DIR=test_infer

python3  test_custom_copy.py \
  --name ${NAME} \
  --nets slbr \
  --models slbr \
  --input-size ${INPUT_SIZE} \
  --crop_size ${INPUT_SIZE} \
  --test-batch 1 \
  --evaluate\
  --preprocess resize \
  --no_flip \
  --mask_mode ${MASK_MODE} \
  --k_center ${K_CENTER} \
  --use_refine \
  --k_refine ${K_REFINE} \
  --k_skip_stage ${K_SKIP} \
  --resume 2021/SLBR-Visible-Watermark-Removal/checkpoint/slbr_v1/checkpoint.pth.tar \
  --test_dir ${TEST_DIR}
  # --checkpoint 2021/SLBR-Visible-Watermark-Removal/checkpoint 
  
