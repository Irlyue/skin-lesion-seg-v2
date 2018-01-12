# tar -zxf ~/models/weights/ssd_mobilenet_v1_coco_11_06_2017.tar.gz -C /tmp/

export MODULE_PATH=/home/wenfeng/models/models/research
export BASE_DIR=/home/wenfeng/all-files/skin-lesion-seg-v2

export PYTHONPATH=$PYTHONPATH:$MODULE_PATH:$MODULE_PATH/slim

rm -r training/train/
#####################################for loop#######################################
for fold in {0..3}
do

# First generate k-th fold data
cd $BASE_DIR
python create_kth_data.py $fold
python create_tfrecord.py

# Then run the training
cd $MODULE_PATH
python object_detection/train.py \
	--logtostderr \
	--pipeline_config_path=$BASE_DIR/training/ssd_mobilenet_v1_pets.config \
	--train_dir=$BASE_DIR/training/train/$fold

done
####################################################################################

exit 0
