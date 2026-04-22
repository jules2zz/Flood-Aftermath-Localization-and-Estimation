
import os
import sys
import shutil
import random
import logging
import multiprocessing
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

import augmentations
import commons
import cosface_loss
import parser
import test
import triplet_loss
import util
from cosplace_model import cosplace_network_stage2
from cosplace_model.cosplace_network_stage2 import ensure_prompt_capacity
from cosplace_model.softmax_loss import CrossEntropyLabelSmooth
from datasets.test_dataset import TestDataset
from datasets.train_dataset import OnlinePairTrainDataset

torch.backends.cudnn.benchmark = True


def build_online_batch(group_ds, args, device, fmi_augmenter, standard_augmenter):
    B = args.batch_size
    num_classes = len(group_ds)
    if num_classes == 0:
        raise ValueError("Current training group has no classes.")

    paired_images = int(round(B * args.paired_batch_ratio)) if args.online_fmi else B
    paired_images = max(2, min(B, paired_images))
    if paired_images % 2 == 1:
        paired_images -= 1
    num_pairs = paired_images // 2
    singleton_images = B - paired_images

    all_classes = list(range(num_classes))
    pair_classes = random.choices(all_classes, k=num_pairs) if num_pairs > num_classes else random.sample(all_classes, num_pairs)
    remaining_classes = [c for c in all_classes if c not in pair_classes]
    if singleton_images > len(remaining_classes):
        singleton_classes = remaining_classes + random.choices(all_classes, k=singleton_images - len(remaining_classes))
    else:
        singleton_classes = random.sample(remaining_classes, singleton_images)

    if args.online_fmi:
        mode_names = ["dry_dry", "dry_flood", "flood_flood"]
        weights = [args.pair_mode_dry_dry, args.pair_mode_dry_flood, args.pair_mode_flood_flood]
        if sum(weights) <= 0:
            raise ValueError("At least one pair-mode weight must be positive.")
    else:
        mode_names = ["dry_dry"]
        weights = [1.0]

    images, targets = [], []
    mode_counter = {"dry_dry": 0, "dry_flood": 0, "flood_flood": 0, "singleton": 0}

    for class_num in pair_classes:
        img1, img2, y, _, _ = group_ds.sample_two_raw_images(class_num)
        pair = torch.stack([img1, img2], dim=0).to(device, non_blocking=True)
        mode = random.choices(mode_names, weights=weights, k=1)[0]
        if args.online_fmi and mode == "dry_flood":
            pair = fmi_augmenter(pair, apply_mask=[False, True])
        elif args.online_fmi and mode == "flood_flood":
            pair = fmi_augmenter(pair, apply_mask=[True, True])
        images.append(pair)
        targets.extend([y, y])
        mode_counter[mode] += 1

    for class_num in singleton_classes:
        img, y, _ = group_ds.sample_raw_image(class_num)
        images.append(img.unsqueeze(0).to(device, non_blocking=True))
        targets.append(y)
        mode_counter["singleton"] += 1

    images = torch.cat(images, dim=0)
    images = standard_augmenter(images)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    return images, targets, mode_counter


cache_path = "./cache"
if os.path.exists(cache_path):
    shutil.rmtree(cache_path)

args = parser.parse_arguments()
start_time = datetime.now()
args.output_folder = f"logs/{args.save_dir}/stage2/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(args.output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")
logging.info("Stage-2 training uses online pair construction with controllable dry/dry, dry/flood, and flood/flood modes.")

groups = [OnlinePairTrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L, current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]
classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold, image_size=args.image_size, resize_test_imgs=args.resize_test_imgs)
test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1", positive_dist_threshold=args.positive_dist_threshold, image_size=args.image_size, resize_test_imgs=args.resize_test_imgs)
logging.info(f"Validation set: {val_ds}")
logging.info(f"Test set: {test_ds}")

model = cosplace_network_stage2.GeoLocalizationNet(args.backbone, args.fc_output_dim, args.train_all_layers)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"模型可更新参数: {name}, 维度: {param.shape}")

prompt_learners = torch.load(args.prompt_learners)
prompt_learners = [prompt_learner.cuda() for prompt_learner in prompt_learners]

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)

model = model.to(args.device).train()
criterion = torch.nn.CrossEntropyLoss()
model_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

if args.resume_train:
    model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = util.resume_train(args, args.output_folder, model, model_optimizer, classifiers, classifiers_optimizers)
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
else:
    best_val_recall1 = start_epoch_num = 0

logging.info("Start training ...")
logging.info(f"There are {len(groups[0])} classes for the first group, each epoch has {args.iterations_per_epoch} iterations with batch_size {args.batch_size}, therefore the model sees each class (on average) {args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")

standard_augmenter = T.Compose([
    augmentations.DeviceAgnosticColorJitter(brightness=args.brightness, contrast=args.contrast, saturation=args.saturation, hue=args.hue),
    augmentations.DeviceAgnosticRandomResizedCrop([args.image_size, args.image_size], scale=[1 - args.random_resized_crop, 1]),
    augmentations.BatchNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

fmi_augmenter = None
if args.online_fmi:
    fmi_augmenter = augmentations.OnlineFMIAugment(
        water_dir=args.fmi_water_dir,
        use_segformer=args.fmi_use_segformer,
        segformer_device=args.fmi_segformer_device,
        water_level_min=args.fmi_water_level_min,
        water_level_max=args.fmi_water_level_max,
        alpha_min=args.fmi_alpha_min,
        alpha_max=args.fmi_alpha_max,
        reflection_strength=args.fmi_reflection_strength,
        edge_preserve=args.fmi_edge_preserve,
        wave_strength=args.fmi_wave_strength,
        noise_std=args.fmi_noise_std,
        dilate_px=(args.fmi_dilate_min, args.fmi_dilate_max),
        water_mix_range=(args.fmi_water_mix_min, args.fmi_water_mix_max),
        horizon_range=(args.fmi_horizon_min, args.fmi_horizon_max),
    ).to(args.device)

if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()

logging.info('start training stage2')
text_features_list_all = []
for g in range(len(groups)):
    batch = args.batch_size
    num_classes = len(groups[g])
    i_ter = num_classes // batch
    left = num_classes - batch * (num_classes // batch)
    if left != 0:
        i_ter = i_ter + 1
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i + 1 != i_ter:
                l_list = torch.arange(i * batch, (i + 1) * batch)
            else:
                l_list = torch.arange(i * batch, num_classes)
            idx = g % len(prompt_learners)
            ensure_prompt_capacity(prompt_learners[idx], l_list)
            text_feature = model(prompt_learner=prompt_learners[idx], label=l_list, get_text=True)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0)
    text_features_list_all.append(text_features)
del prompt_learners

if torch.cuda.device_count() > 1:
    logging.info(f"[Info] Using {torch.cuda.device_count()} GPUs via DataParallel")
    model = torch.nn.DataParallel(model)

for epoch_num in range(start_epoch_num, args.epochs_num):
    epoch_start_time = datetime.now()
    current_group_num = epoch_num % args.groups_num
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)
    text_features_list = text_features_list_all[current_group_num].to(args.device)
    loss_cross = CrossEntropyLabelSmooth(num_classes=len(groups[current_group_num]))
    model = model.train()

    epoch_losses = np.zeros((0, 1), dtype=np.float32)
    mode_stats = {"dry_dry": 0, "dry_flood": 0, "flood_flood": 0, "singleton": 0}
    for _ in tqdm(range(args.iterations_per_epoch), ncols=100):
        images, targets, iter_mode_counter = build_online_batch(groups[current_group_num], args, args.device, fmi_augmenter, standard_augmenter)
        for k, v in iter_mode_counter.items():
            mode_stats[k] += v

        model_optimizer.zero_grad()
        classifiers_optimizers[current_group_num].zero_grad()

        if not args.use_amp16:
            image_features = model(images)
            output = classifiers[current_group_num](image_features, targets)
            loss = criterion(output, targets)
            logits = image_features @ text_features_list.t()
            i2tloss = loss_cross(logits, targets)
            loss = loss + i2tloss
            if args.soft_triplet:
                tripletloss = triplet_loss.triplet_loss(image_features, targets, margin=0.2, norm_feat=True, hard_mining=True) * args.batch_size
                loss = loss + tripletloss
            loss.backward()
            epoch_losses = np.append(epoch_losses, loss.item())
            del loss, output
            model_optimizer.step()
            classifiers_optimizers[current_group_num].step()
        else:
            with torch.cuda.amp.autocast():
                image_features = model(images)
                output = classifiers[current_group_num](image_features, targets)
                loss = criterion(output, targets)
                logits = image_features @ text_features_list.t()
                i2tloss = loss_cross(logits, targets)
                loss = loss + i2tloss
                if args.soft_triplet:
                    tripletloss = triplet_loss.triplet_loss(image_features, targets, margin=0.2, norm_feat=True, hard_mining=True) * args.batch_size
                    loss = loss + tripletloss
            scaler.scale(loss).backward()
            epoch_losses = np.append(epoch_losses, loss.item())
            del loss, output
            scaler.step(model_optimizer)
            scaler.step(classifiers_optimizers[current_group_num])
            scaler.update()

    classifiers[current_group_num] = classifiers[current_group_num].cpu()
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
    text_features_list = text_features_list.cpu()

    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, loss = {epoch_losses.mean():.4f}")
    logging.info(f"Epoch {epoch_num:02d} mode counts: {mode_stats}")

    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
    is_best = recalls[0] > best_val_recall1
    best_val_recall1 = max(recalls[0], best_val_recall1)
    util.save_checkpoint({
        "epoch_num": epoch_num + 1,
        "model_state_dict": (model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
    }, is_best, args.output_folder)

    if (epoch_num + 1) % args.checkpoint_period_stage2 == 0:
        torch.save((model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()), f"{args.output_folder}/model_{epoch_num+1}.pth")

logging.info(f"Trained for stage2 {args.epochs_num:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

best_model_state_dict = torch.load(f"{args.output_folder}/best_model.pth")
if isinstance(model, torch.nn.DataParallel):
    best_model_state_dict = {f"module.{k}": v for k, v in best_model_state_dict.items()}
model.load_state_dict(best_model_state_dict)

logging.info(f"Now testing on the test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model, args.num_preds_to_save)
logging.info(f"{test_ds}: {recalls_str}")
logging.info("Experiment finished (without any errors)")
