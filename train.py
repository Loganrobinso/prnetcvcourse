import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import monai
import sys
from torchmetrics.classification import JaccardIndex
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from prnet.utils import (
    load_prototype_features,
    get_formatted_cst_time,
    global_config_logger,
    setup_logger,
)
from prnet.datasets.mvtec import TRAINMVTEC, MVTEC
from prnet.models.prnet import PRNet
from prnet.losses.focal_loss import FocalLoss
from prnet.losses.smooth_l1_loss import SmoothL1Loss
from test import validate
from config import Config

def main(c, class_name):
    save_dir = '/app/training_plots'
    logger = setup_logger('./logs/training_logger')

    train_dataset = TRAINMVTEC(
        c.config_dic["dataset"]["data_path"],
        c.config_dic["dataset"]["anomaly_source_path"],
        class_name=class_name,
        train=True,
        img_size=256,
        crp_size=256,
        msk_size=256,
        msk_crp_size=256,
        num_anomalies=c.config_dic["num_anomalies"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=c.config_dic["batch_size"],
        shuffle=True,
        num_workers=0, # for now, leave it like this
        drop_last=True,
    )

    test_dataset = MVTEC(
        c.config_dic["dataset"]["data_path"],
        class_name=class_name,
        train=False,
        img_size=256,
        crp_size=256,
        msk_size=256,
        msk_crp_size=256,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=c.config_dic["batch_size"],
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PRNet(c.config_dic["model"]["model_name"], num_classes=30, device=device).to(
        device
    )

    model = nn.DataParallel(model, device_ids=[0, 1])

    proto_features = load_prototype_features(
        c.config_dic["dataset"]["prototype_path"], class_name, device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=c.config_dic["optimizer"]["learning_rate"],
        weight_decay=c.config_dic["optimizer"]["weight_decay"],
    )

    dice_focal_loss = monai.losses.GeneralizedDiceFocalLoss(
        to_onehot_y=True, softmax=True, reduction='mean', weight=None,
        gamma=c.config_dic["loss_functions"]["focal_gamma"], 
        lambda_focal=c.config_dic["loss_functions"]["focal_alpha"] 
    )

    cce = nn.CrossEntropyLoss()

    training_miou = JaccardIndex(task='multiclass', threshold=0.5, num_classes=30, average='macro').to(device=device)
    per_class_training_miou = JaccardIndex(task='multiclass', threshold=0.5, num_classes=30, average=None).to(device=device)

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10, T_mult=2, eta_min=1e-6, last_epoch=-1
    )

    best_val_miou, miou_epoch = 0, 0
    best_jaccard, jaccard_epoch = 0, 0
    import gc

    for epoch in range(c.config_dic["epochs"]):
        torch.cuda.empty_cache()
        gc.collect()
        model.train()
        train_loss_total, total_num = 0, 0
        progress_bar = tqdm(total=len(train_loader))
        progress_bar.set_description(f"Epoch[{epoch}/{c.config_dic['epochs']}]")
        jaccard_list = []

        for batch_idx, batch in enumerate(train_loader):
            progress_bar.update(1)
            images, temp_labels, mask = batch

            images = images.to(device)
            mask = mask.to(device)

            logits = model(images, proto_features)

            temp_mask = mask.squeeze(1)

            df_loss = dice_focal_loss(logits, mask)
            cce_loss = cce(logits, temp_mask)

            loss = df_loss + cce_loss
            
            training_miou.update(logits, temp_mask)
            per_class_training_miou.update(logits, temp_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            total_num += 1

            batch_idx_step = batch_idx / len(train_loader)
            scheduler.step(epoch + batch_idx_step)
        
        progress_bar.close()
        scheduler.step()
        train_loss = train_loss_total / total_num
        train_miou = training_miou.compute().cpu().numpy()
        pc_miou = per_class_training_miou.compute().cpu().numpy()

        pcm_values = [round(miou_value, 4) for miou_value in pc_miou]

        logger.info(
            f"\nEpoch [{epoch}/{c.config_dic['epochs']}]: | Train Loss: {train_loss}\
            \nTrain MIoU: {train_miou:.4f} | Per-Class MIoU: {pcm_values}\n"
        )

        if (c.config_dic["eval_freq"] > 0) and (
            (epoch + 1) % c.config_dic["eval_freq"] == 0
        ):
            val_miou, val_pc_miou = validate(model, test_loader, proto_features, device, epoch)
            
            logger.info(
                f"\nEpoch: {epoch} | Class Name: {class_name}\
                    \nVal MIoU: {val_miou} | Val Per-Class MIoU: {val_pc_miou}\n"
            )

            os.makedirs(
                os.path.join(c.config_dic["timestamped_checkpoint_dir"], class_name),
                exist_ok=True,
            )
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                ckpt_path = os.path.join(
                    c.config_dic["timestamped_checkpoint_dir"],
                    class_name,
                    c.config_dic["model"]["model_name"] + "best-miou.pth",
                )
                torch.save(model.state_dict(), ckpt_path)
                miou_epoch = epoch

    return best_val_miou, miou_epoch


if __name__ == "__main__":
    # Argument Parser init
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file_path",
        required=True,
        type=str,
        help="Get the path to the config file.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Setup the config
    c = Config(args.config_file_path)

    # Setup timestamped checkpoint dir
    timestamp = get_formatted_cst_time()
    c.config_dic["timestamped_checkpoint_dir"] = os.path.join(
        c.config_dic["checkpoint_dir"], timestamp
    )

    # Setup logger
    global_config_logger(
        log_file=os.path.join(c.config_dic["log_dir"], timestamp + ".log")
    )
   # logger = setup_logger(__name__)

    # Determine classes to train on
    if c.config_dic["class_name"] == "all":
        class_names = MVTEC.CLASS_NAMES
    else:
        class_names = [c.config_dic["class_name"]]

    for class_name in class_names:
        best_val_miou, miou_epoch = main(c, class_name)
        logger.info(f"{class_name}: Best, MIoU: {best_val_miou} Epoch: {miou_epoch}")
