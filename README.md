# PairedGen: Image-Mask Paired Generation for Improving Polyp Segmentation

### News

2025/2/25 Update the code.

### Requirements

```shell
conda create -n PairedGen python=3.8
conda activate PairedGen

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.27.2

pip install datasets==2.19.1
pip install transformers accelerate xformers==0.0.20 wandb
pip install bitsandbytes

pip install tensorboard==2.12.0
pip install tensorboardX
pip install opencv-python
```

We use the SD1.5 pre-training model.

### Dataset

We conduct extensive experiments on five polyp segmentation datasets
following [PraNet](https://github.com/DengPingFan/PraNet). The traing images are stored in the `images/` directory, and the traing masks are placed in the `conditioning_images/` directory. The text prompt is just "a sks endoscope image" for image, and "polyp" for mask. 

The `data.py` file should be named to match the current directory `data`, and the paths within it may should be changed to absolute paths.

```
data/
|--conditioning_images/
|	|--1.png
|	|--2.png
|--images/
|	|--1.png
|	|--2.png
|--data.py
|--prompt.json
```

Data [HQSeg44k](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/data) for stage 1 pre-training, we only use its training data, and the prompt file will be uploaded later.

### Training 

We used `train_oursnet.py` to train our framework. 

First, configure the GPUs. 

```
accelerate config
```

And then: 

stage 1:

```shell
export MODEL_DIR="pre-training_SD_1.5_path"
export DATASET_NAME="./data_HQSEG44K"
export OUTPUT_DIR="checkpoints_save_path"

accelerate launch train_oursnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=$DATASET_NAME \
 --resolution=256 \
 --learning_rate=1e-5 \
 --validation_prompt="In the image, there is a stone chair placed in front of a hedge or shrubbery.  &&&  chair" \
 --train_batch_size=1 \
 --num_train_epochs=500 \
 --gradient_accumulation_steps=8 \
 --checkpointing_steps=500 \
 --mixed_precision="fp16" --use_8bit_adam
```

stage 2:

```shell
export MODEL_DIR="pre-training_SD_1.5_path"
export DATASET_NAME="./data"
export OUTPUT_DIR="checkpoints_save_path"

accelerate launch train_oursnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=$DATASET_NAME \
 --controlnet_model_name_or_path="pre-training_oursnet_path_in_HQSEG44k" \
 --resolution=256 \
 --learning_rate=1e-5 \
 --validation_prompt="a sks endoscope image. &&& Polyp." \
 --train_batch_size=1 \
 --num_train_epochs=500 \
 --gradient_accumulation_steps=8 \
 --checkpointing_steps=500 \
 --mixed_precision="fp16" --use_8bit_adam
```

### Inference

We used `infer.py` to generate several images and masks for inference and `infer_pipe.py` to generate data equivalent to the original training set. Need to update the `base_model_path` and `controlnet_path`, as well as the save path.

```shell
python infer.py
python infer_pipe.py
```

### Downstream Evaluation

To perform downstream evaluation, follow  `python infer_pipe.py` to sample image-mask pairs and mix them with the original training dataset, the model for polyp segmentation:

- [PraNet](https://github.com/DengPingFan/PraNet)
- [SANet](https://github.com/weijun88/SANet)
- [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT)

### Checkpoints

TODO

### Citation
```
TODO
```
