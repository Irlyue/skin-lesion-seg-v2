export MODULE_PATH=/home/wenfeng/models/models/research
export BASE_DIR=/home/wenfeng/all-files/skin-lesion-seg-v2
export PYTHONPATH=$PYTHONPATH:$MODULE_PATH:$MODULE_PATH/slim

#####################################for loop#######################################
for fold in 0 1 2 3
do

echo "-------------------------->Evaluating model $fold<---------------------------"

python eval_kfold.py $fold 0.5

done
####################################################################################

exit 0
