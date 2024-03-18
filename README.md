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

## File Description
```
project
│   README.md 
│
└───baseline
│   │   baseline_concat.py
│   │   baseline_concat.sh(run file for baseline)
│   │
│   └───model_baseline_ckpt(checkpoints file for baseline testing)
│   
└───dataset-process
│   └───MELD(features)
│   │   |   dev/train/test_a_3dim.csv(3 dimensional features from audio)
│   │   |   dev/train/test_meld_au_text_i.csv(semantic descriptions for visual modality)
│   │   |   dev/train/test_trans.csv(utterances)
│   │   |   dev/train/test_mix_sentence.csv(combination for semantic description of all modalities(text+audio+visual))
│   │   |   dev/train/test_mix_at/vt_sentence.csv(combination for semantic description of two modalities(at:text+audio vt:visual+text))
│   │   |   outputIS09_dev/train/test.csv(extracted features from audio by OpenSmile(IS09))
│   │   |   meld_label_3way.npz(generated label file)
│   │
│   └───features/MELD(features for baseline)
│   │   |   openface_meld.tar.gz(extracted features by OpenFace)(big files!)
│   │   |
│   │   └───audio(audio features extracted by OpenSmile)
│   │   |
│   │   └───text(textual features extracted by Sentence-Transformer)
│   │   |
│   │   └───visual(visual features extracted by ImageNet)
│   │
└───dataset-release(original label files for MELD）
│   │
└───data(wav files for MELD dataset)
│   └───audio_split(audio files for dataset)
│   │
└───MELD_result(testing result)
│   
└───feature_extract(feature extraction files)
│   │   config.py
│   │   extract_imagenet_embedding.py(visual feature extraction)
│   │   extract_visual_features.sh(run file for extracting visual features)
│   │   audio_process.py((audio feature extraction))
│   |
└───prompt(run files for prompting)
│   │   meld_test.sh(run file for test)
│   │   meld_val.sh(run file for training the validation)
│   │   meld_test.py
│   │   meld_val.py
│   │   Temps.py(label mapping groups)
│   │
│   └───dataset
│   │   └───dataloader.py
│   │
│   └───model_ckpt(checkpoint files)
│   │
│   └───models(pretrained model)

```

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
Or you could use the extracted features in the corresponding files in this project.

## Run

You can run the code with the following example:

'''
cd ./prompt
#Training and Validation
sh meld_val.sh
#When testing, you should set the checkpoint file name when running the test file
sh meld_test.sh $checkpointfile_name

'''

If you want the baseline result, run the code:

'''
cd ./baseline
sh baseline_concat.sh
'''
