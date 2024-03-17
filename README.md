# Project

## Links

* [Overview](#overview)
* [Requirements](#requirements)
* [Prepare the data](#prepare-the-data)
* [Run the model](#run-lm-bff)
  * [Quick start](#quick-start)
  * [Experiments with multiple runs](#experiments-with-multiple-runs)
  * [Using demonstrations with filtering](#using-demonstrations-with-filtering)
  * [Automatically searched prompt](#automatically-searched-prompt)
  * [Ensemble](#ensemble-model)
  * [Zero-shot experiments](#zero-shot-experiments)
  * [How to design your own templates](#how-to-design-your-own-templates)
* [Citation](#citation)

## Requirements

To run the code, please install all the dependency packages:

- Python 3.8
- CUDA 10.2
- pytorch ==1.8.0
- torchvision == 0.9.0
- fairseq == 0.10.1
- transformers==4.5.1
- pandas == 1.2.5
- wenetruntime
- paddlespeech == 1.4.1

If you find some problems in running, please check the environment file in detail: env.yaml

## Prepare the data

The original MELD dataset is offered [here](https://affective-meld.github.io/). You can download it from the website.

## Run

### Quick start
Our code is built on [transformers](https://github.com/huggingface/transformers) and we use its `3.4.0` version. Other versions of `transformers` might cause unexpected errors.

Before running any experiments, create the result folder by `mkdir result` to save checkpoints. Then you can run our code with the following example:

```bash
python run.py \
    --task_name SST-2 \
    --data_dir data/k-shot/SST-2/16-42 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --model_name_or_path roberta-large \
    --few_shot_type prompt-demo \
    --num_k 16 \
    --max_steps 1000 \
    --eval_steps 100 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 0 \
    --output_dir result/tmp \
    --seed 42 \
    --template "*cls**sent_0*_It_was*mask*.*sep+*" \
    --mapping "{'0':'terrible','1':'great'}" \
    --num_sample 16 \
```
