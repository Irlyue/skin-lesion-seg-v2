export MODULE_PATH=/home/wenfeng/models/models/research
export BASE_DIR=/home/wenfeng/all-files/skin-lesion-seg-v2

cd $MODULE_PATH
pwd

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo $PATH
echo $PYTHONPATH

python object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=$BASE_DIR/training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix=$BASE_DIR/training/train/model.ckpt-2224 \
    --output_directory=$BASE_DIR/training/output_inference_graph.pb
