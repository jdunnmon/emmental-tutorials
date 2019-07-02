import logging

import argparse
from functools import partial
import sys
import torch.nn as nn
import torch.nn.functional as F

import emmental
from cxr_dataset import CXR8Dataset
from emmental import Meta
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from modules.classification_module import ClassificationModule
from modules.torch_vision_encoder import TorchVisionEncoder
from task_config import CXR8_TASK_NAMES
from transforms import get_data_transforms
from emmental.utils.parse_arg import str2bool
sys.path.append('../superglue')
from utils import write_to_file

# Initializing logs
logger = logging.getLogger(__name__)
emmental.init("logs")

# Defining helper functions
def ce_loss(task_name, immediate_ouput, Y, active):
    return F.cross_entropy(
        immediate_ouput[f"classification_module_{task_name}"][0], Y.view(-1) - 1
    )

def output(task_name, immediate_ouput):
    return F.softmax(immediate_ouput[f"classification_module_{task_name}"][0], dim=1)

def parse_args():
    parser = argparse.ArgumentParser(description='Run chexnet slicing experiments')
    parser.add_argument('--data_name', default='CXR8', help='Dataset name')
    parser.add_argument('--cxrdata_path', 
                        default=f"/dfs/scratch1/senwu/mmtl/emmental-tutorials/chexnet/data/nih_labels.csv",
                        help='Path to labels')
    parser.add_argument('--cxrimage_path', 
                        default=f"/dfs/scratch1/senwu/mmtl/emmental-tutorials/chexnet/data/images",
                        help='Path to images')
    parser.add_argument('--tasks', default='CXR8', type=str, nargs='+',
                        help='list of tasks; if CXR8, all CXR8; if TRIAGE, normal/abnormal')
    parser.add_argument(
        "--slices", type=str2bool, default=False, help="Whether to include slices"
    )
    parser.add_argument(
        "--train", type=str2bool, default=True, help="Whether to train the model"
    )
    args = parser.parse_args()
    return args

if __name__=="__main__":
    # Parsing command line arguments
    args = parse_args()
    
    # Ensure that global state is fresh
    #Meta.reset()
        
    # Initialize Emmental
    #config = parse_arg_to_config(args)
    ## HACK: handle None in model_path, proper way to handle this in the
    ## next release of Emmental
    #if (
    #    config["model_config"]["model_path"]
    #    and config["model_config"]["model_path"].lower() == "none"
    #):
    #    config["model_config"]["model_path"] = None
    #emmental.init(config["meta_config"]["log_path"], config=config)
    
    # Configuring run data
    Meta.update_config(
        config={
            "meta_config": {"seed": 1701, "device": 0},
            "learner_config": {
                "n_epochs": 2,
                "valid_split": "val",
                "optimizer_config": {"optimizer": "sgd", "lr": 0.001, "l2": 0.000},
                "lr_scheduler_config": {
                    "warmup_steps": None,
                    "warmup_unit": "batch",
                    "lr_scheduler": "linear",
                    "min_lr": 1e-6,
                },
            },
            "logging_config": {
                "counter_unit": "epoch",
                "evaluation_freq": 1,
                "writer_config": {"writer": "tensorboard", "verbose": True},
                "checkpointing": True,
                "checkpointer_config": {
                    "checkpoint_path": None,
                    "checkpoint_freq": 1,
                    "checkpoint_metric": {f"model/val/all/loss": "min"},
                    "checkpoint_task_metrics": {"model/train/all/loss": "min"},
                    "checkpoint_runway": 0,
                    "checkpoint_clear": True,
                },
            },
        },
    )

    # Save command line argument into file
    cmd_msg = " ".join(sys.argv)
    logger.info(f"COMMAND: {cmd_msg}")
    write_to_file(Meta.log_path, "cmd.txt", cmd_msg)
    
    #Meta.config["learner_config"]["global_evaluation_metric_dict"] = {
    #    f"model/SuperGLUE/{split}/score": partial(superglue_scorer, split=split)
    #    for split in ["val"]
    #}
    
    # Save Emmental config into file
    logger.info(f"Config: {Meta.config}")
    write_to_file(Meta.log_path, "config.txt", Meta.config)
    
    # Getting paths to data
    DATA_NAME = args.data_name
    CXRDATA_PATH = args.cxrdata_path
    CXRIMAGE_PATH = args.cxrimage_path

    # Providing model settings
    BATCH_SIZE = 16
    CNN_ENCODER = "densenet121"

    BATCH_SIZES = {"train": 16, "val": 64, "test": 64}

    # Getting transforms
    cxr8_transform = get_data_transforms(DATA_NAME)

    # Getting task to label dict
    # All possible tasks in dataloader
    all_tasks = CXR8_TASK_NAMES +['Abnormal']
    if 'CXR8' in args.tasks:
        # Standard chexnet
        logging.info('Using all CXR8 tasks')
        task_list = CXR8_TASK_NAMES
        add_binary_triage_label=False
    elif 'TRIAGE' in args.tasks:
        # Binary triage
        logging.info('Using only Abnormal task')
        task_list = ['Abnormal']
        add_binary_triage_label=True
    else:
        # Otherwise, making sure tasks are valid
        logging.info('Using only specified tasks')
        task_list = args.tasks
        for task in task_list:
            assert(task in all_tasks)
        add_binary_triage_label=True

    task_to_label_dict = {task_name: task_name for task_name in task_list}
    print(task_to_label_dict)

    # Creating datasets
    datasets = {}

    for split in ["train", "val", "test"]:

        datasets[split] = CXR8Dataset(
            name=DATA_NAME,
            path_to_images=CXRIMAGE_PATH,
            path_to_labels=CXRDATA_PATH,
            split=split,
            transform=cxr8_transform[split],
            sample=0,
            seed=1701,
            add_binary_triage_label=add_binary_triage_label,
        )

        logger.info(f"Loaded {split} split for {DATA_NAME}.")

    # Building dataloaders
    dataloaders = []

    for split in ["train", "val", "test"]:
        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict=task_to_label_dict,
                dataset=datasets[split],
                split=split,
                shuffle=True if split == "train" else False,
                batch_size=BATCH_SIZES[split],
                num_workers=8,
            )
        )
        #if args.slices:
        #    logger.info("Initializing task-specific slices")
        #    slice_func_dict = slicing.slice_func_dict[task_name]
        #    # Include general purpose slices
        #    if args.general_slices:
        #        logger.info("Including general slices")
        #        slice_func_dict.update(slicing.slice_func_dict["general"])

        #    task_dataloaders = slicing.add_slice_labels(
        #        task_name, task_dataloaders, slice_func_dict
        #    )

        #    slice_tasks = slicing.add_slice_tasks(
        #        task_name, task, slice_func_dict, args.slice_hidden_dim
        #    )
        #    tasks.extend(slice_tasks)
        logger.info(f"Built dataloader for {datasets[split].name} {split} set.")

    # Building Emmental tasks
    input_shape = (3, 224, 224)

    # Load pretrained model if necessary
    cnn_module = TorchVisionEncoder(CNN_ENCODER, pretrained=True)
    classification_layer_dim = cnn_module.get_frm_output_size(input_shape)

    tasks = [
        EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "cnn": cnn_module,
                    f"classification_module_{task_name}": ClassificationModule(
                        classification_layer_dim, 2
                    ),
                }
            ),
            task_flow=[
                {"name": "cnn", "module": "cnn", "inputs": [("_input_", "image")]},
                {
                    "name": f"classification_module_{task_name}",
                    "module": f"classification_module_{task_name}",
                    "inputs": [("cnn", 0)],
                },
            ],
            loss_func=partial(ce_loss, task_name),
            output_func=partial(output, task_name),
            scorer=Scorer(metrics=["roc_auc", "f1", "precision", "recall", "accuracy"]),
        )
        for task_name in task_list
    ]

    # Defining model and trainer
    model = EmmentalModel(name="Chexnet", tasks=tasks)
    
    if args.train:
        # Training model
        emmental_learner = EmmentalLearner()
        emmental_learner.learn(model, dataloaders)
        
    # If model is slice-aware, slice scores will be calculated from slice heads
    # If model is not slice-aware, manually calculate performance on slices
    if not args.slices:
        slice_func_dict = {}
        slice_keys = args.task
        if args.general_slices:
            slice_keys.append("general")

        for k in slice_keys:
            slice_func_dict.update(slicing.slice_func_dict[k])

        scores = slicing.score_slices(model, dataloaders, args.task, slice_func_dict)
    else:
        scores = model.score(dataloaders)
            
    # Save metrics into file
    logger.info(f"Metrics: {scores}")
    write_to_file(Meta.log_path, "metrics.txt", scores)
    
    
    # Save best metrics into file
    if args.train:
        logger.info(
            f"Best metrics: "
            f"{emmental_learner.logging_manager.checkpointer.best_metric_dict}"
        )
        write_to_file(
            Meta.log_path,
            "best_metrics.txt",
            emmental_learner.logging_manager.checkpointer.best_metric_dict,
        )