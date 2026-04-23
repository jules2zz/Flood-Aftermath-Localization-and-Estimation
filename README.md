# Flood-Aftermath-Localization-and-Estimation
Leveraging visual foundation models for urban flooding localization and impact estimation from social media
FALE: Flood Aftermath Localization and Estimation
Official codebase for a two-part framework for image-based urban flood analysis:
FAL: Flood Aftermath Localizer, a street-level geolocation model for flood-related imagery under severe appearance shift.
FAE: Flood Aftermath Estimator, a visual-language flood impact assessment model fine-tuned with LoRA to predict multi-axis aftermath severity codes.
This repository is organized around the full pipeline:
Where? Localize a flood image to its most likely street location using large-scale street-view retrieval.
How severe? Estimate visible aftermath impacts from the localized image using a fine-tuned vision-language model.
<p align="center">
  <img src="assets/overview.png" width="95%" alt="FALE overview">
</p>
---
Highlights
Two-stage Flood Aftermath Localizer (FAL) tailored to flood-induced appearance changes.
Geo-Prompt Forge for place-aware visual-text alignment.
Vision Anchor Refine with online Flood-Masked Invariance (FMI) augmentation.
Flood Aftermath Estimator (FAE) based on LoRA fine-tuning of a pretrained VLM.
Retrieval evaluation with Recall@K and qualitative retrieval visualization.
Multi-axis flood impact estimation across flood depth, human impact, property damage, building damage, infrastructure damage, economic disruption, and environmental debris.
---
Repository structure
The current code is split into two modules:
```text
.
├── FAL/
│   ├── train\_stage1.py
│   ├── train\_stage2.py
│   ├── eval.py
│   ├── test.py
│   ├── parser.py
│   ├── augmentations.py
│   ├── commons.py
│   ├── util.py
│   ├── visualizations.py
│   ├── cosface\_loss.py
│   ├── triplet\_loss.py
│   ├── clip/
│   ├── cosplace\_model/
│   └── datasets/
└── FAE/
    ├── run\_flood\_sft.py
    ├── GFA\_train\_85.jsonl
    ├── GFA\_test\_15.jsonl
    └── dataset\_info.json
```
Recommended asset layout for the public GitHub repository:
```text
.
├── assets/
│   ├── overview.png
│   ├── FAL.png
│   ├── FAE.png
│   └── result1.png
├── FAL/
├── FAE/
├── README.md
├── requirements.txt
└── LICENSE
```
---
Method overview
FAL: Flood Aftermath Localizer
FAL is designed for street-level localization of flood aftermath imagery. Instead of assuming clean visual overlap between query and reference images, it explicitly addresses the appearance shift caused by floodwater, reflections, splashing, debris, and transient scene changes.
The training pipeline contains two stages.
Stage 1: Geo-Prompt Forge
Stage 1 learns a place-aware prompt space using frozen CLIP image and text representations. The goal is to align each geographic class with a learnable textual prompt, so that the model captures stable place semantics rather than overfitting to superficial appearance.
<p align="center">
  <img src="assets/FAL.png" width="95%" alt="FAL training framework">
</p>
Stage 2: Vision Anchor Refine
Stage 2 refines the image encoder for robust place recognition. It introduces:
visual cue anchoring, using prompt-derived text features as semantic references;
classification and metric learning objectives, including cosine-margin classification and optional triplet loss;
online Flood-Masked Invariance (FMI), which synthesizes flood-like perturbations and enforces invariance between dry and flood views from the same location.
This stage is the core mechanism that improves retrieval robustness under flood-induced domain shift.
Retrieval output
Given a flood image query, FAL retrieves the most likely street-view locations from a large georeferenced database and evaluates them using Recall@K.
<p align="center">
  <img src="assets/result1.png" width="95%" alt="Retrieval examples">
</p>
Green borders indicate correct retrievals, and red borders indicate failure cases.
---
FAE: Flood Aftermath Estimator
FAE is a visual-language flood impact assessor built on top of a pretrained VLM with LoRA fine-tuning. It predicts visible aftermath categories and their severity using a compact flood-specific coding system.
The current implementation uses a structured instruction-following setup with codes such as:
`FD`: Flood Depth
`HU`: Human Impact
`PD`: Property Damage
`BD`: Building Damage
`IN`: Infrastructure Damage
`EC`: Economic Disruption
`EN`: Environmental Debris & Sediment
Each axis is scored on a 3-level severity scale, for example `FD2`, `IN3`, or `EN1`.
<p align="center">
  <img src="assets/FAE.png" width="95%" alt="FAE fine-tuning framework">
</p>
The estimator is intended to answer the question: what visible impacts are present in the image, and how severe are they?
---
Installation
1. Create the Python environment for FAL
```bash
conda create -n fale python=3.10 -y
conda activate fale
pip install -r FAL/requirements.txt
pip install torch torchvision tqdm utm scikit-learn pillow faiss-cpu
```
If you want GPU FAISS, replace `faiss-cpu` with the appropriate GPU build for your environment.
2. Additional dependencies for FAE
FAE uses `llamafactory-cli` and a pretrained Qwen2.5-VL checkpoint in the current training script. A typical setup is:
```bash
conda create -n fae python=3.10 -y
conda activate fae
pip install llamafactory
pip install transformers datasets accelerate peft pillow numpy
```
You should also make sure the base model path in `FAE/run\_flood\_sft.py` points to your local checkpoint:
```python
model\_name\_or\_path="/path/to/Qwen2.5-VL-7B-Instruct"
```
---
Data preparation
FAL data format
The FAL code expects image filenames to encode geographic metadata in the file path, following the CosPlace-style convention used by the dataset loader.
Training images
Training classes are constructed from image paths containing UTM and heading information. In `datasets/train\_dataset.py`, class ids and groups are built from filename tokens extracted using `path.split("@")`.
The code expects image paths to contain at least:
UTM east at token index `1`
UTM north at token index `2`
heading at token index `9`
A typical file naming convention therefore looks like:
```text
some\_prefix@utm\_east@utm\_north@...@heading@image.jpg
```
Test / evaluation images
For testing, the loader in `datasets/test\_dataset.py` expects:
database images in `database/`
query images in `queries/` or `queries\_v1/`
UTM east and north embedded in the filename
Example layout:
```text
DATA\_ROOT/
├── train/
├── val/
│   ├── database/
│   └── queries/
└── test/
    ├── database/
    ├── queries/
    └── queries\_v1/
```
The default positive match threshold is controlled by:
```bash
--positive\_dist\_threshold 25
```
which means a retrieval is considered correct if it falls within 25 meters of a positive reference.
FMI texture data
If you enable online FMI in Stage 2, you must provide a folder of water textures:
```bash
--online\_fmi --fmi\_water\_dir /path/to/water\_textures
```
Optional SegFormer-based road masking can also be enabled with:
```bash
--fmi\_use\_segformer
```
FAE data format
The FAE training and test sets are currently stored as JSONL files in ShareGPT-style multimodal format.
Each example contains:
`images`: a relative image path
`messages`: a conversation with `system`, `user`, and `assistant` turns
Example:
```json
{
  "images": \["flood\_image/Bing\_0001 (10).jpeg"],
  "messages": \[
    {"role": "system", "content": "..."},
    {"role": "user", "image": 0, "content": "<image> Identify visible flood impacts..."},
    {"role": "assistant", "content": "EN3"}
  ]
}
```
In the provided code:
`GFA\_train\_85.jsonl` is the training split
`GFA\_test\_15.jsonl` is the test split
`dataset\_info.json` stores LLaMA-Factory dataset registration metadata
---
Quick start
FAL stage 1 training
Stage 1 learns the prompt representations while keeping the visual encoder frozen.
```bash
cd FAL
CUDA\_VISIBLE\_DEVICES=0 python train\_stage1.py \\
  --backbone ViT-B-16 \\
  --fc\_output\_dim 512 \\
  --batch\_size\_stage1 512 \\
  --epochs\_num\_stage1 480 \\
  --train\_set\_folder /path/to/train \\
  --test\_set\_folder /path/to/test \\
  --save\_dir fal\_vitb16\_stage1
```
Important notes:
`--test\_set\_folder` is still required by the shared parser, even for training.
The main output of Stage 1 is a saved prompt learner file such as:
```text
logs/<save\_dir>/stage1/<timestamp>/last\_prompt\_learners.pth
```
FAL stage 2 training
Stage 2 refines the image encoder using classification, image-text alignment, and optional triplet loss.
```bash
cd FAL
CUDA\_VISIBLE\_DEVICES=0 python train\_stage2.py \\
  --backbone ViT-B-16 \\
  --fc\_output\_dim 512 \\
  --batch\_size 32 \\
  --epochs\_num 64 \\
  --iterations\_per\_epoch 10000 \\
  --lr 1e-5 \\
  --classifiers\_lr 1e-2 \\
  --train\_set\_folder /path/to/train \\
  --val\_set\_folder /path/to/val \\
  --test\_set\_folder /path/to/test \\
  --prompt\_learners /path/to/last\_prompt\_learners.pth \\
  --save\_dir fal\_vitb16\_stage2 \\
  --use\_amp16 \\
  --soft\_triplet
```
FAL stage 2 with online FMI
```bash
cd FAL
CUDA\_VISIBLE\_DEVICES=0 python train\_stage2.py \\
  --backbone ViT-B-16 \\
  --fc\_output\_dim 512 \\
  --batch\_size 32 \\
  --epochs\_num 64 \\
  --iterations\_per\_epoch 10000 \\
  --train\_set\_folder /path/to/train \\
  --val\_set\_folder /path/to/val \\
  --test\_set\_folder /path/to/test \\
  --prompt\_learners /path/to/last\_prompt\_learners.pth \\
  --save\_dir fal\_vitb16\_fmi \\
  --use\_amp16 \\
  --soft\_triplet \\
  --online\_fmi \\
  --paired\_batch\_ratio 0.5 \\
  --pair\_mode\_dry\_dry 0.1 \\
  --pair\_mode\_dry\_flood 0.8 \\
  --pair\_mode\_flood\_flood 0.1 \\
  --fmi\_water\_dir /path/to/water\_textures
```
The FMI-related arguments exposed in `parser.py` let you control water level, blending strength, reflection strength, edge preservation, wave perturbation, and mask dilation.
FAL evaluation
```bash
cd FAL
CUDA\_VISIBLE\_DEVICES=0 python eval.py \\
  --backbone ViT-B-16 \\
  --fc\_output\_dim 512 \\
  --resume\_model /path/to/best\_model.pth \\
  --test\_set\_folder /path/to/test \\
  --infer\_batch\_size 64 \\
  --positive\_dist\_threshold 25
```
To save qualitative predictions:
```bash
cd FAL
CUDA\_VISIBLE\_DEVICES=0 python eval.py \\
  --backbone ViT-B-16 \\
  --fc\_output\_dim 512 \\
  --resume\_model /path/to/best\_model.pth \\
  --test\_set\_folder /path/to/test \\
  --num\_preds\_to\_save 3
```
This will report:
`R@1`
`R@5`
`R@10`
`R@20`
and optionally save qualitative retrieval results.
FAE fine-tuning
The current FAE script performs stage-wise LoRA fine-tuning and evaluation.
In the provided code:
base model: `Qwen2.5-VL-7B-Instruct`
finetuning type: `LoRA`
total epochs: `36`
epochs per stage: `3`
batch size per device: `2`
gradient accumulation steps: `8`
learning rate: `1e-5`
Run:
```bash
cd FAE
python run\_flood\_sft.py
```
The script will:
train for one stage;
locate the latest LoRA adapter directory;
run prediction on the test split;
compute macro and micro precision, recall, and F1;
repeat until all stages finish.
Output is saved under a directory like:
```text
saves/Qwen2.5-VL-7B-Instruct/lora/flood\_<timestamp>/
```
---
Main arguments
FAL
Some of the most important arguments in `FAL/parser.py` are:
Argument	Meaning
`--backbone`	CLIP backbone: `CLIP-RN50`, `CLIP-RN101`, `CLIP-ViT-B-16`, `CLIP-ViT-B-32`
`--fc\_output\_dim`	final descriptor dimension
`--train\_set\_folder`	training image root
`--val\_set\_folder`	validation set root
`--test\_set\_folder`	test set root
`--batch\_size\_stage1`	stage-1 batch size
`--epochs\_num\_stage1`	stage-1 epochs
`--batch\_size`	stage-2 batch size
`--epochs\_num`	stage-2 epochs
`--iterations\_per\_epoch`	stage-2 iterations per epoch
`--prompt\_learners`	path to saved stage-1 prompt learners
`--online\_fmi`	enable online FMI synthesis
`--fmi\_water\_dir`	folder of water textures
`--soft\_triplet`	enable triplet loss
`--positive\_dist\_threshold`	positive radius in meters
`--num\_preds\_to\_save`	number of retrieved predictions to save
FAE
The current `run\_flood\_sft.py` exposes its main hyperparameters as in-file constants and a `BASE\_ARGS` dictionary rather than CLI flags. The ones you are most likely to change are:
`model\_name\_or\_path`
`dataset`
`val\_size`
`learning\_rate`
`per\_device\_train\_batch\_size`
`gradient\_accumulation\_steps`
`lora\_rank`
`lora\_alpha`
`freeze\_vision\_tower`
`freeze\_multi\_modal\_projector`
`TOTAL\_EPOCHS`
`EPOCHS\_PER\_STAGE`
`EVAL\_DATASET`
---
Outputs
FAL outputs
Stage 1 output:
```text
logs/<save\_dir>/stage1/<timestamp>/
├── last\_prompt\_learners.pth
└── prompt\_learners\_<epoch>.pth
```
Stage 2 output:
```text
logs/<save\_dir>/stage2/<timestamp>/
├── best\_model.pth
├── last\_checkpoint.pth
├── model\_<epoch>.pth
└── predictions/...
```
FAE outputs
```text
saves/Qwen2.5-VL-7B-Instruct/lora/flood\_<timestamp>/
├── epoch\_3/
├── epoch\_6/
├── ...
└── train.log
```
Each stage directory contains LoRA adapters and a `predict\_test/` directory with `generated\_predictions.jsonl` after inference.
---
Reproducing the full FALE pipeline
A typical workflow is:
Prepare the large-scale street-view reference dataset for FAL.
Train FAL Stage 1 to obtain prompt learners.
Train FAL Stage 2 with or without FMI for flood-robust descriptors.
Evaluate retrieval performance using Recall@K and save qualitative examples.
Prepare the GFA JSONL data for FAE.
Fine-tune the VLM with LoRA using the flood aftermath coding scheme.
Run FAE inference and evaluation to obtain code-level prediction quality.
Combine the two modules so that a real-world flood image can first be localized and then assessed.
---
Practical notes
The FAL parser is shared across training and evaluation, so some arguments that look unnecessary may still be required.
Stage 1 and Stage 2 currently rely on saved prompt learners as the bridge between the two stages.
FAL assumes a CosPlace-style filename metadata convention. If your data naming differs, update the parsing logic in `datasets/train\_dataset.py` and `datasets/test\_dataset.py`.
FAE currently assumes a local LLaMA-Factory workflow and a local Qwen2.5-VL checkpoint.
If you release this publicly, add:
a `requirements\_fae.txt` or `environment.yml`;
a small `scripts/` folder with reproducible shell commands;
public dataset access instructions;
pretrained checkpoint download links.
---
Suggested GitHub cleanup before release
Before making the repository public, it is worth cleaning up a few things:
add a top-level `.gitignore`;
move the figures into `assets/`;
add `requirements\_fae.txt` or a unified environment file;
remove `\_\_pycache\_\_` and notebook checkpoint folders;
rename dataset paths and hard-coded local paths into configurable arguments;
add `scripts/train\_fal\_stage1.sh`, `scripts/train\_fal\_stage2.sh`, `scripts/eval\_fal.sh`, and `scripts/train\_fae.sh`;
add a `Model Zoo` section when checkpoints are ready.
---
Citation
If you use this codebase, please cite the corresponding paper once available:
```bibtex
@article{your\_fale\_paper,
  title   = {FALE: Flood Aftermath Localization and Estimation from Images},
  author  = {Author, A. and Author, B. and Author, C.},
  journal = {arXiv or Journal Name},
  year    = {2026}
}
```
You may also cite the localization and estimation modules separately if they are released as independent papers.
---
Acknowledgements
This codebase builds on ideas from large-scale visual place recognition, CLIP-based representation learning, and LoRA-based multimodal fine-tuning. The public release can acknowledge upstream projects that inspired the implementation and engineering structure.
---
Contact
For questions, issues, or collaboration, please open a GitHub issue or contact the corresponding author.
