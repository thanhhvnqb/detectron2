# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
PoseNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import itertools
import json
import copy

from fvcore.common.file_io import PathManager
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.utils.logger import setup_logger

from posenet import add_posenet_config

COCODEV = {}
COCODEV["coco"] = {
    "coco_2017_testdev": (
        "coco/test2017",
        "coco/annotations/image_info_test-dev2017.json",
    ),
    "coco_2017_test": (
        "coco/test2017",
        "coco/annotations/image_info_test2017.json",
    ),
}

class COCODevEvaluator(COCOEvaluator):
    def _eval_predictions(self, tasks):
        """
        Evaluate self._predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            json.encoder.FLOAT_REPR = lambda x: format(x, '.2f')
            json.encoder.LONG_REPR = lambda x: format(x, '.2l')
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            out_results = copy.deepcopy(self._coco_results)
            for result in out_results:
                result['score'] = round(result['score'], 4)
                result['bbox'] = [round(x) for x in result['bbox']]
                result['keypoints'] = [round(x) for x in result['keypoints']]
                result['keypoints'][2::3] = [1 for _ in range(len(result['keypoints'])//3)]
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(out_results))
                f.flush()

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_" + dataset_name)
        if dataset_name in COCODEV["coco"].keys():
            return COCODevEvaluator(dataset_name, cfg, True, output_folder)
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def register_cocodev(root="datasets"):
    for dataset_name, splits_per_dataset in COCODEV.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_posenet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="posenet")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        register_cocodev()
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Train with FCOS detector, #conv=1; CONV for KP.")
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
