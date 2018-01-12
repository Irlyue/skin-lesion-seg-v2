export MODULE_PATH=/home/wenfeng/models/models/research
export BASE_DIR=/home/wenfeng/all-files/skin-lesion-seg-v2
export PYTHONPATH=$PYTHONPATH:$MODULE_PATH:$MODULE_PATH/slim

cd $MODULE_PATH

#####################################for loop#######################################
for fold in {0..3}
do

python object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=$BASE_DIR/training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix=$BASE_DIR/training/train/$fold/model.ckpt-2000 \
    --output_directory=$BASE_DIR/training/train/$fold/output_inference_graph.pb

done
####################################################################################

exit 0
