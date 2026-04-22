
import argparse


def parse_arguments(is_training: bool = True):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--M", type=int, default=10, help="_")
    parser.add_argument("--alpha", type=int, default=30, help="_")
    parser.add_argument("--N", type=int, default=5, help="_")
    parser.add_argument("--L", type=int, default=2, help="_")
    parser.add_argument("--groups_num", type=int, default=8, help="_")
    parser.add_argument("--min_images_per_class", type=int, default=10, help="_")

    parser.add_argument(
        "--backbone",
        type=str,
        default="CLIP-RN50",
        choices=["CLIP-RN50", "CLIP-RN101", "CLIP-ViT-B-16", "CLIP-ViT-B-32", "RN50", "RN101", "ViT-B-16", "ViT-B-32"],
        help="_",
    )
    parser.add_argument("--fc_output_dim", type=int, default=1024,
                        help="Output dimension of final fully connected layer")
    parser.add_argument("--train_all_layers", default=False, action="store_true",
                        help="If true, train all layers of the backbone")

    parser.add_argument("--use_amp16", action="store_true", help="use Automatic Mixed Precision")
    parser.add_argument("--augmentation_device", type=str, default="cuda", choices=["cuda", "cpu"], help="on which device to run data augmentation")
    parser.add_argument("--batch_size", type=int, default=32, help="_")
    parser.add_argument("--epochs_num", type=int, default=64, help="_")
    parser.add_argument("--iterations_per_epoch", type=int, default=10000, help="_")
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--classifiers_lr", type=float, default=0.01, help="_")
    parser.add_argument("--image_size", type=int, default=512, help="Width and height of training images")
    parser.add_argument("--resize_test_imgs", default=False, action="store_true",
                        help="Resize test images to image_size along the shorter side while maintaining aspect ratio")

    parser.add_argument("--batch_size_stage1", type=int, default=512, help="_")
    parser.add_argument("--epochs_num_stage1", type=int, default=480, help="_")
    parser.add_argument("--lr_stage1", type=float, default=0.01, help="_")
    parser.add_argument("--resume_model_stage1", type=str, default=None, help="_")
    parser.add_argument("--checkpoint_period_stage1", type=int, default=80, help="_")
    parser.add_argument("--checkpoint_period_stage2", type=int, default=8, help="_")
    parser.add_argument("--cache_feature_folder", type=str, default=None, help="_")
    parser.add_argument("--prompt_learners", type=str, default=None, help="_")
    parser.add_argument("--soft_triplet", action="store_true", help="_")
    parser.add_argument("--freeze_cnn", type=int, default=2, help="_")
    parser.add_argument("--freeze_trans", type=int, default=6, help="_")

    parser.add_argument("--brightness", type=float, default=0.7, help="_")
    parser.add_argument("--contrast", type=float, default=0.7, help="_")
    parser.add_argument("--hue", type=float, default=0.5, help="_")
    parser.add_argument("--saturation", type=float, default=0.7, help="_")
    parser.add_argument("--random_resized_crop", type=float, default=0.5, help="_")

    parser.add_argument("--online_fmi", action="store_true", help="Enable online FMI flood synthesis during stage-2 training")
    parser.add_argument("--paired_batch_ratio", type=float, default=0.5,
                        help="Fraction of batch images coming from same-class paired subset")
    parser.add_argument("--pair_mode_dry_dry", type=float, default=0.1,
                        help="Sampling weight for dry-dry positive pairs")
    parser.add_argument("--pair_mode_dry_flood", type=float, default=0.8,
                        help="Sampling weight for dry-flood positive pairs")
    parser.add_argument("--pair_mode_flood_flood", type=float, default=0.1,
                        help="Sampling weight for flood-flood positive pairs")
    parser.add_argument("--fmi_water_level_min", type=float, default=0.50,
                        help="Minimum normalized waterline position for online FMI")
    parser.add_argument("--fmi_water_level_max", type=float, default=0.85,
                        help="Maximum normalized waterline position for online FMI")
    parser.add_argument("--fmi_alpha_min", type=float, default=0.25,
                        help="Minimum blending strength for online FMI")
    parser.add_argument("--fmi_alpha_max", type=float, default=0.55,
                        help="Maximum blending strength for online FMI")
    parser.add_argument("--fmi_reflection_strength", type=float, default=0.20,
                        help="Reflection contribution for online FMI")
    parser.add_argument("--fmi_edge_preserve", type=float, default=0.20,
                        help="Edge-preservation strength for online FMI")
    parser.add_argument("--fmi_wave_strength", type=float, default=0.04,
                        help="Wave perturbation strength for online FMI")
    parser.add_argument("--fmi_water_dir", type=str, default=None,
                        help="Directory containing water textures for online FMI")
    parser.add_argument("--fmi_use_segformer", action="store_true",
                        help="Use SegFormer road masking for online FMI if available")
    parser.add_argument("--fmi_segformer_device", type=str, default=None,
                        help="Device for SegFormer masking, e.g. cuda or cpu")
    parser.add_argument("--fmi_noise_std", type=float, default=0.05, help="Alpha-mask noise amplitude")
    parser.add_argument("--fmi_dilate_min", type=int, default=5, help="Minimum mask dilation size")
    parser.add_argument("--fmi_dilate_max", type=int, default=11, help="Maximum mask dilation size")
    parser.add_argument("--fmi_water_mix_min", type=float, default=0.70, help="Minimum water/reflection mixing weight")
    parser.add_argument("--fmi_water_mix_max", type=float, default=1.00, help="Maximum water/reflection mixing weight")
    parser.add_argument("--fmi_horizon_min", type=float, default=0.35, help="Minimum reflection horizon ratio")
    parser.add_argument("--fmi_horizon_max", type=float, default=0.65, help="Maximum reflection horizon ratio")

    parser.add_argument("--infer_batch_size", type=int, default=64,
                        help="Batch size for inference (validating and testing)")
    parser.add_argument("--positive_dist_threshold", type=int, default=25,
                        help="distance in meters for a prediction to be considered a positive")
    parser.add_argument("--resume_train", type=str, default=None,
                        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth")
    parser.add_argument("--resume_model", type=str, default=None,
                        help="path to model to resume, e.g. logs/.../best_model.pth")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("--num_workers", type=int, default=8, help="_")
    parser.add_argument("--num_preds_to_save", type=int, default=0,
                        help="At the end of training, save N preds for each query")
    parser.add_argument("--save_only_wrong_preds", action="store_true",
                        help="When saving preds save only difficult queries")
    if is_training:
        parser.add_argument("--train_set_folder", type=str, help="path of the folder with training images")
        parser.add_argument("--val_set_folder", type=str, help="path of the folder with val images")
    parser.add_argument("--test_set_folder", type=str, required=True,
                        help="path of the folder with test images")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="name of directory on which to save the logs, under logs/save_dir")

    args = parser.parse_args()

    backbone_aliases = {
        "RN50": "CLIP-RN50",
        "RN101": "CLIP-RN101",
        "ViT-B-16": "CLIP-ViT-B-16",
        "ViT-B-32": "CLIP-ViT-B-32",
    }
    args.backbone = backbone_aliases.get(args.backbone, args.backbone)

    if args.paired_batch_ratio <= 0 or args.paired_batch_ratio > 1:
        parser.error("--paired_batch_ratio must be in (0, 1].")
    if args.online_fmi:
        if not args.fmi_water_dir:
            parser.error("--online_fmi requires --fmi_water_dir.")
        if args.pair_mode_dry_dry + args.pair_mode_dry_flood + args.pair_mode_flood_flood <= 0:
            parser.error("At least one pair mode weight must be positive.")
    if args.fmi_use_segformer and not args.online_fmi:
        parser.error("--fmi_use_segformer requires --online_fmi.")
    return args
