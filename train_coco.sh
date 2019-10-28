# export Dataset=/home/codepro/ThanhHV/Dataset/mscoco
# ln -s $Dataset/annotations datasets/coco/annotations
# ln -s $Dataset/train2017 datasets/coco/train2017
# ln -s $Dataset/test2017 datasets/coco/test2017
# ln -s $Dataset/val2017 datasets/coco/val2017

rpnfunc ()
{
    local date=20191017
    local netname=rpn
    # local netname=rpn_iouloss

    python -m torch.distributed.launch\
    --nproc_per_node=2\
    --master_port=$((RANDOM + 10000))\
    tools/train_net.py --config-file configs/rpn_R_50_FPN_1x.yaml\
    DATALOADER.NUM_WORKERS 2\
    OUTPUT_DIR ./out/$netname/$date/ 2>&1 | tee -a "./out/run_${date}_$netname.log"
}

maskrcnnfunc ()
{
    local date=20191020
    # local netname=maskrcnn
    local netname=maskrcnn_iouloss

    python -m torch.distributed.launch\
    --nproc_per_node=2\
    --master_port=$((RANDOM + 10000))\
    tools/train_net.py --config-file configs/e2e_mask_rcnn_R_50_FPN_1x.yaml\
    DATALOADER.NUM_WORKERS 2\
    OUTPUT_DIR ./out/$netname/$date/ 2>&1 | tee -a "./out/run_${date}_$netname.log"
}

kprcnnfunc ()
{
    local date=20191020
    local netname=kprcnn
    # local netname=kprcnn_iouloss

    python tools/train_net.py --num-gpus 2\
        --config-file configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml\
        OUTPUT_DIR ./out/$netname/$date/ 2>&1 | tee -a "./out/run_${date}_$netname.log"
}

kprcnnfunc
# rpnfunc
# maskrcnnfunc
# fcosfunc
