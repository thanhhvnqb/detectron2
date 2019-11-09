#!/bin/bash
# export Dataset=/home/codepro/ThanhHV/Dataset/mscoco
# export Dataset=/home/thanh/DATA/Dataset/human_pose/mscoco/
# ln -s $Dataset/annotations datasets/coco/annotations
# ln -s $Dataset/train2017 datasets/coco/train2017
# ln -s $Dataset/test2017 datasets/coco/test2017
# ln -s $Dataset/val2017 datasets/coco/val2017

# netname='kprcnn'
netname='kprcnn_fcos'
# netname='posenet_fcos_conv'
# netname='posenet_fcos_fc'
# netname='posenet_rcnn_fc'
run_date=20191102
test_conf=test
outdir=../../out/$netname/$run_date/
outlog=../../out/run_test_${run_date}_$netname.log
if [ "$netname" = 'kprcnn' ];
then
    configfile=configs/kprcnn_R_50_FPN_$test_conf.yaml
elif [ "$netname" = 'kprcnn_fcos' ];
then
    configfile=configs/kprcnn_R_50_fcos_FPN_$test_conf.yaml
elif [ "$netname" = 'posenet_fcos_conv' ];
then
    configfile=configs/posenet_fcos_conv_R_50_FPN_$test_conf.yaml
elif [ "$netname" = 'posenet_fcos_fc' ];
then
    configfile=configs/posenet_fcos_fc_R_50_FPN_$test_conf.yaml
elif [ "$netname" = 'posenet_rcnn_fc' ];
then
    configfile=configs/posenet_rcnn_fc_R_50_FPN_$test_conf.yaml
fi

if test -f "$outlog"; then
    rm $outlog
fi
CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 --resume --eval-only --config-file $configfile OUTPUT_DIR $outdir 2>&1 | tee -a $outlog

