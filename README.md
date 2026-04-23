# FALE: Flood Aftermath Localization and Estimation

Official codebase for a two-part framework for image-based urban flood analysis:

- **FAL**: Flood Aftermath Localizer, a street-level geolocation model for flood-related imagery under severe appearance shift.
- **FAE**: Flood Aftermath Estimator, a visual-language flood impact assessment model fine-tuned with LoRA to predict multi-axis aftermath severity codes.

This repo is organized around the full pipeline:

1. **Where?** Localize a flood image to its most likely street location.
2. **How severe?** Estimate visible aftermath impacts from the localized image.

<p align="center">
  <img src="overview.png" width="95%" alt="FALE overview">
</p>

---


## Repository structure

The current code is split into two modules:

```text
.
├── FAL/
│   ├── train_stage1.py
│   ├── train_stage2.py
│   ├── eval.py
│   ├── test.py
│   ├── parser.py
│   ├── augmentations.py
│   ├── commons.py
│   ├── util.py
│   ├── visualizations.py
│   ├── cosface_loss.py
│   ├── triplet_loss.py
│   ├── clip/
│   ├── cosplace_model/
│   └── datasets/
└── FAE/
    ├── run_flood_sft.py
    ├── GFA_train_85.jsonl
    ├── GFA_test_15.jsonl
    └── dataset_info.json
```

## FAL: Flood Aftermath Localizer

FAL is designed for **street-level localization of flood aftermath imagery**. Instead of assuming clean visual overlap between query and reference images, it explicitly addresses the appearance shift caused by floodwater, reflections, splashing, debris, and transient scene changes.

The training pipeline contains two stages.

### Stage 1: Geo-Prompt Forge

Stage 1 learns a **place-aware prompt space** using frozen CLIP image and text representations. The goal is to align each geographic class with a learnable textual prompt, so that the model captures stable place semantics rather than overfitting to superficial appearance.

<p align="center">
  <img src="FAL.png" width="95%" alt="FAL training framework">
</p>

### Stage 2: Vision Anchor Refine

Stage 2 refines the image encoder for robust place recognition. It introduces:

- **visual cue anchoring**, using prompt-derived text features as semantic references;
- **classification and metric learning objectives**, including cosine-margin classification and triplet loss;
- **online Flood-Masked Invariance (FMI)**, which synthesizes flood-like perturbations and enforces invariance between dry and flood views from the same location.

This stage is the core mechanism that improves retrieval robustness under flood-induced domain shift.

### Retrieval output

Given a flood image query, FAL retrieves the most likely street-view locations from a large georeferenced database and evaluates them using **Recall@K**.

<p align="center">
  <img src="result1.png" width="95%" alt="Retrieval examples">
</p>

Green borders indicate correct retrievals, and red borders indicate failure cases.

---

## FAE: Flood Aftermath Estimator

FAE is a **visual-language flood impact assessor** built on top of a pretrained VLM with **LoRA fine-tuning**. It predicts visible aftermath categories and their severity using a compact flood-specific coding system.

The current implementation uses a structured instruction-following setup with codes such as:

- `FD`: Flood Depth
- `HU`: Human Impact
- `PD`: Property Damage
- `BD`: Building Damage
- `IN`: Infrastructure Damage
- `EC`: Economic Disruption
- `EN`: Environmental Debris & Sediment

Each axis is scored on a 3-level severity scale, for example `FD2`, `IN3`, or `EN1`.

<p align="center">
  <img src="FAE.png" width="95%" alt="FAE fine-tuning framework">
</p>

The estimator is intended to answer the question: **what visible impacts are present in the image, and how severe are they?**

---

## Installation

## 1. Create the Python environment for FAL

```bash
pip install -r FAL/requirements.txt
pip install torch torchvision tqdm utm scikit-learn pillow faiss-cpu
```

If you want GPU FAISS, replace `faiss-cpu` with the appropriate GPU build for your environment.

## 2. Additional dependencies for FAE

FAE uses `llamafactory-cli` and a pretrained Qwen2.5-VL checkpoint in the current training script. A typical setup is:

```bash
pip install llamafactory
```

You should also make sure the base model path in `FAE/run_flood_sft.py` points to your local checkpoint:

```python
model_name_or_path="/path/to/Qwen2.5-VL-7B-Instruct"
```

---

## Data preparation

## FAL data format

The FAL code expects image filenames to encode geographic metadata in the file path, following the CosPlace-style convention used by the dataset loader. 
`@ UTM_east @ UTM_north @ UTM_zone_number @ UTM_zone_letter @ latitude @ longitude @ pano_id @ @ heading @ @ @ @ timestamp @ @.jpg`

For example an image can be named: `@0544388.51@4172758.25@10@S@037.70098@-122.49646@kPnABhWJ1kA61eMNPLs1vQ@@210@@@@201606@@.jpg`

### Test / evaluation images

For testing, the loader in `datasets/test_dataset.py` expects:

- database images in `database/`
- query images in `queries/` or `queries_v1/`
- UTM east and north embedded in the filename

Example layout:

```text
DATA_ROOT/
├── train/
├── val/
│   ├── database/
│   └── queries/
└── test/
    ├── database/
    ├── queries/
    └── queries_v1/
```

The default positive match threshold is controlled by:

```bash
--positive_dist_threshold 50
```

which means a retrieval is considered correct if it falls within 50 meters of a positive reference. For real world flood management, this distance is suitable for municipalities to react.

## FMI texture data

If you enable online FMI in Stage 2, you must provide a folder of water textures:

```bash
--online_fmi --fmi_water_dir /path/to/water_textures
```

Optional SegFormer-based road masking can also be enabled with:

```bash
--fmi_use_segformer
```

## FAE data format

The FAE training and test sets are currently stored as JSONL files in **ShareGPT-style multimodal format**.

Each example contains:

- `images`: a relative image path
- `messages`: a conversation with `system`, `user`, and `assistant` turns

Example:

```json
{
  "images": ["flood_image/Bing_0001 (10).jpeg"],
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "image": 0, "content": "<image> Identify visible flood impacts..."},
    {"role": "assistant", "content": "EN3"}
  ]
}
```

In the provided code:

- `GFA_train_85.jsonl` is the training split
- `GFA_test_15.jsonl` is the test split
- `dataset_info.json` stores LLaMA-Factory dataset registration metadata

---

## Quick start

## FAL stage 1 training

Stage 1 learns the prompt representations while keeping the visual encoder frozen.

```bash
cd FAL
CUDA_VISIBLE_DEVICES=0 python train_stage1.py \
  --backbone ViT-B-16 \
  --fc_output_dim 512 \
  --batch_size_stage1 512 \
  --epochs_num_stage1 480 \
  --train_set_folder /path/to/train \
  --test_set_folder /path/to/test \
  --save_dir fal_vitb16_stage1
```

Important notes:

- `--test_set_folder` is still required by the shared parser, even for training.
- The main output of Stage 1 is a saved prompt learner file such as:

```text
logs/<save_dir>/stage1/<timestamp>/last_prompt_learners.pth
```

## FAL stage 2 training

Stage 2 refines the image encoder using classification, image-text alignment, and optional triplet loss.

```bash
cd FAL
CUDA_VISIBLE_DEVICES=0 python train_stage2.py \
  --backbone ViT-B-16 \
  --fc_output_dim 512 \
  --batch_size 32 \
  --epochs_num 64 \
  --iterations_per_epoch 10000 \
  --train_set_folder /path/to/train \
  --val_set_folder /path/to/val \
  --test_set_folder /path/to/test \
  --prompt_learners /path/to/last_prompt_learners.pth \
  --save_dir fal_vitb16_fmi \
  --use_amp16 \
  --soft_triplet \
  --online_fmi \
  --paired_batch_ratio 0.5 \
  --pair_mode_dry_dry 0.1 \
  --pair_mode_dry_flood 0.8 \
  --pair_mode_flood_flood 0.1 \
  --fmi_water_dir /path/to/water_textures
```

The FMI-related arguments exposed in `parser.py` let you control water level, blending strength, reflection strength, edge preservation, wave perturbation, and mask dilation.

## FAL evaluation

```bash
cd FAL
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --backbone ViT-B-16 \
  --resume_model /path/to/best_model.pth \
  --test_set_folder /path/to/test \
  --positive_dist_threshold 50 \
  --num_preds_to_save 5
```

This will report:

- `R@1`
- `R@5`
- `R@10`
- `R@20`

and save qualitative retrieval results.

## FAE fine-tuning

The current FAE script performs stage-wise LoRA fine-tuning and evaluation.

In the provided code:

- base model: `Qwen2.5-VL-7B-Instruct`
- finetuning type: `LoRA`
- total epochs: `36`
- epochs per stage: `3`
- batch size per device: `2`
- gradient accumulation steps: `8`
- learning rate: `1e-5`

Run:

```bash
cd FAE
python run_flood_sft.py
```

The script will:

1. train for one stage;
2. locate the latest LoRA adapter directory;
3. run prediction on the test split;
4. compute macro and micro precision, recall, and F1;
5. repeat until all stages finish.

Output is saved under a directory like:

```text
saves/Qwen2.5-VL-7B-Instruct/lora/flood_<timestamp>/
```

---

## Main arguments

## FAL

Some of the most important arguments in `FAL/parser.py` are:

| Argument | Meaning |
|---|---|
| `--backbone` | CLIP backbone: `CLIP-RN50`, `CLIP-RN101`, `CLIP-ViT-B-16`, `CLIP-ViT-B-32` |
| `--fc_output_dim` | final descriptor dimension |
| `--train_set_folder` | training image root |
| `--val_set_folder` | validation set root |
| `--test_set_folder` | test set root |
| `--batch_size_stage1` | stage-1 batch size |
| `--epochs_num_stage1` | stage-1 epochs |
| `--batch_size` | stage-2 batch size |
| `--epochs_num` | stage-2 epochs |
| `--iterations_per_epoch` | stage-2 iterations per epoch |
| `--prompt_learners` | path to saved stage-1 prompt learners |
| `--online_fmi` | enable online FMI synthesis |
| `--fmi_water_dir` | folder of water textures |
| `--soft_triplet` | enable triplet loss |
| `--positive_dist_threshold` | positive radius in meters |
| `--num_preds_to_save` | number of retrieved predictions to save |

## FAE

The current `run_flood_sft.py` exposes its main hyperparameters as in-file constants and a `BASE_ARGS` dictionary rather than CLI flags. The ones you are most likely to change are:

- `model_name_or_path`
- `dataset`
- `val_size`
- `learning_rate`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `lora_rank`
- `lora_alpha`
- `freeze_vision_tower`
- `freeze_multi_modal_projector`
- `TOTAL_EPOCHS`
- `EPOCHS_PER_STAGE`
- `EVAL_DATASET`

---

## Acknowledgements

This codebase builds on ideas from large-scale visual place recognition, CLIP-based representation learning, and LoRA-based multimodal fine-tuning. The public release can acknowledge upstream projects that inspired the implementation and engineering structure.

* This work is inspired by [ProGEO](https://github.com/Chain-Mao/ProGEO).
* Parts of the code are based on [CosPlace](https://github.com/gmberton/CosPlace).
* The FAE module was trained using the [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) framework.

---

