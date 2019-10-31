# export Dataset=/home/codepro/ThanhHV/Dataset/mscoco
# export Dataset=/home/thanh/DATA/Dataset/human_pose/mscoco/
# ln -s $Dataset/annotations datasets/coco/annotations
# ln -s $Dataset/train2017 datasets/coco/train2017
# ln -s $Dataset/test2017 datasets/coco/test2017
# ln -s $Dataset/val2017 datasets/coco/val2017

kprcnnmodfunc ()
{
    local date=20191030
    local netname=kprcnn_mod

    python train_net.py --num-gpus 2\
        --config-file configs/kprcnn_R_50_FPN_1x.yaml\
        OUTPUT_DIR ../../out/$netname/$date/ 2>&1 | tee -a "../../out/run_${date}_$netname.log"
}

kprcnnmodfunc
