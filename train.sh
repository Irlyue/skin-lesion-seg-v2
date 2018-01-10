tar -zxf ~/models/weights/ssd_mobilenet_v1_coco_11_06_2017.tar.gz -C /tmp/

export MODULE_PATH=/home/wenfeng/models/models/research
export BASE_DIR=/home/wenfeng/all-files/skin-lesion-seg-v2
#python create_csv.py
#python create_tfrecord.py

cd $MODULE_PATH
pwd

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo $PATH
echo $PYTHONPATH

python object_detection/train.py \
	--logtostderr \
	--pipeline_config_path=$BASE_DIR/training/ssd_mobilenet_v1_pets.config \
	--train_dir=$BASE_DIR/training/train/


