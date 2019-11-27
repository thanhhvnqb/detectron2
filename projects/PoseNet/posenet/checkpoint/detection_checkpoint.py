# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pickle
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm

from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts


class DevDetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in detectron & detectron2
    model zoo, and apply conversions for legacy models.
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )

    def _load_file(self, filename):
        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            if not self.has_checkpoint():
                loaded = dict(("backbone.bottom_up." + key, value) for (key, value) in loaded.items())
            loaded = {"model": loaded}
        return loaded

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            model_state_dict = self.model.state_dict()
            align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint["model"] = model_state_dict
        # for non-caffe2 models, use standard ways to load it
        super()._load_model(checkpoint)
