#!/bin/bash
# export Dataset=/home/codepro/ThanhHV/Dataset/mscoco
# export Dataset=/home/thanh/DATA/Dataset/human_pose/mscoco/
# ln -s $Dataset/annotations datasets/coco/annotations
# ln -s $Dataset/train2017 datasets/coco/train2017
# ln -s $Dataset/test2017 datasets/coco/test2017
# ln -s $Dataset/val2017 datasets/coco/val2017

netname='kprcnn_fcos'
# netname='posenet_fcos_conv'
run_date=$(date +%Y%m%d)
# run_date=201910930
# test_conf=instant_test
test_conf=1x
outdir=out/$netname/$run_date/
outlog=out/run_${run_date}_$netname.log
if [ "$netname" = 'kprcnn_fcos' ];
then
    configfile=configs/kprcnn_R_50_fcos_FPN_$test_conf.yaml
elif [ "$netname" = 'posenet_fcos_conv' ];
then
    configfile=configs/posenet_fcos_conv_R_50_FPN_$test_conf.yaml
fi

if test -f "$outlog"; then
    rm $outlog
fi
python train_net.py --num-gpus 2 --config-file $configfile OUTPUT_DIR $outdir 2>&1 | tee -a $outlog

