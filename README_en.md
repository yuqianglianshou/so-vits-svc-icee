<div align="center">
<img alt="LOGO" src="https://avatars.githubusercontent.com/u/127122328?s=400&u=5395a98a4f945a3a50cb0cc96c2747505d190dbc&v=4" width="300" height="300" />
  
# SoftVC VITS Singing Voice Conversion

[**English**](./README_en.md) | [**中文简体**](./README.md)

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/svc-develop-team/so-vits-svc/blob/4.1-Stable/docs/notebooks/sovits4_for_colab.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-AGPL3.0-green.svg?style=for-the-badge)](https://github.com/svc-develop-team/so-vits-svc/blob/4.1-Stable/LICENSE)

This round of limited time update is coming to an end, the warehouse will enter the Archieve state, please know

</div>

> ✨ A studio that contains a visible f0 editor and other Onnx-based features: [MoeVoiceStudio](https://github.com/NaruseMioShirakana/MoeVoiceStudio)

> ✨ A fork with a greatly improved user interface: [34j/so-vits-svc-fork](https://github.com/34j/so-vits-svc-fork)

> ✨ A client supports real-time conversion: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

**This project differs fundamentally from VITS, as it focuses on Singing Voice Conversion (SVC) rather than Text-to-Speech (TTS). In this project, TTS functionality is not supported, and VITS is incapable of performing SVC tasks. It's important to note that the models used in these two projects are not interchangeable or universally applicable.**

## Announcement

The purpose of this project was to enable developers to have their beloved anime characters perform singing tasks. The developers' intention was to focus solely on fictional characters and avoid any involvement of real individuals, anything related to real individuals deviates from the developer's original intention.

## Disclaimer

This project is an open-source, offline endeavor, and all members of SvcDevelopTeam, as well as other developers and maintainers involved (hereinafter referred to as contributors), have no control over the project. The contributors have never provided any form of assistance to any organization or individual, including but not limited to dataset extraction, dataset processing, computing support, training support, inference, and so on. The contributors do not and cannot be aware of the purposes for which users utilize the project. Therefore, any AI models and synthesized audio produced through the training of this project are unrelated to the contributors. Any issues or consequences arising from their use are the sole responsibility of the user.

This project is run completely offline and does not collect any user information or gather user input data. Therefore, contributors to this project are not aware of all user input and models and therefore are not responsible for any user input.

This project serves as a framework only and does not possess speech synthesis functionality by itself. All functionalities require users to train the models independently. Furthermore, this project does not come bundled with any models, and any secondary distributed projects are independent of the contributors of this project.

## 📏 Terms of Use

# Warning: Please ensure that you address any authorization issues related to the dataset on your own. You bear full responsibility for any problems arising from the usage of non-authorized datasets for training, as well as any resulting consequences. The repository and its maintainer, svc develop team, disclaim any association with or liability for the consequences. 

1. This project is exclusively established for academic purposes, aiming to facilitate communication and learning. It is not intended for deployment in production environments.
2. Any sovits-based video posted to a video platform must clearly specify in the introduction the input source vocals and audio used for the voice changer conversion, e.g., if you use someone else's video/audio and convert it by separating the vocals as the input source, you must give a clear link to the original video or music; if you use your own vocals or a voice synthesized by another voice synthesis engine as the input source, you must also state this in your introduction.
3. You are solely responsible for any infringement issues caused by the input source and all consequences. When using other commercial vocal synthesis software as an input source, please ensure that you comply with the regulations of that software, noting that the regulations of many vocal synthesis engines explicitly state that they cannot be used to convert input sources!
4. Engaging in illegal activities, as well as religious and political activities, is strictly prohibited when using this project. The project developers vehemently oppose the aforementioned activities. If you disagree with this provision, the usage of the project is prohibited.
5. If you continue to use the program, you will be deemed to have agreed to the terms and conditions set forth in README and README has discouraged you and is not responsible for any subsequent problems.
6. If you intend to employ this project for any other purposes, kindly contact and inform the maintainers of this repository in advance.

## 📝 Model Introduction

The singing voice conversion model uses SoftVC content encoder to extract speech features from the source audio. These feature vectors are directly fed into VITS without the need for conversion to a text-based intermediate representation. As a result, the pitch and intonations of the original audio are preserved. Meanwhile, the vocoder was replaced with [NSF HiFiGAN](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan) to solve the problem of sound interruption.

## 🖥️ Platform and Hardware Notes

The most reliable current statement is:

- Verified:
  - `Windows 11 + NVIDIA GPU (12GB)` works for both training and inference
- Theoretically viable:
  - `Linux + NVIDIA GPU` should be workable for the main-model training flow, inference, and app entrypoints
- Actual code-level constraints today:
  - main-model training requires `NVIDIA GPU + CUDA`
  - diffusion training should also be treated as an `NVIDIA GPU + CUDA` workflow
  - main-model training supports single-node multi-GPU
  - diffusion training is currently designed as single-GPU
  - `macOS` should not be advertised as a training platform; it is more appropriate for inference attempts or non-training checks

So if you plan to use this repository outside `Windows + NVIDIA GPU`, treat it as "reviewed in code and theoretically viable", not "fully hardware-validated by the repository".

### 🆕 4.1-Stable Version Update Content

- Feature input is changed to the 12th Layer of [Content Vec](https://github.com/auspicious3000/contentvec) Transformer output, And compatible with 4.0 branches.
- Update the shallow diffusion, you can use the shallow diffusion model to improve the sound quality.
- Added Whisper-PPG encoder support
- Added static/dynamic sound fusion
- Added loudness embedding
- Added Functionality of feature retrieval from [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
  
### 🆕 Questions about compatibility with the 4.0 model

- To support the 4.0 model and incorporate the speech encoder, you can make modifications to the `config.json` file. Add the `speech_encoder` field to the "model" section as shown below:

```
  "model": {
    .........
    "ssl_dim": 256,
    "n_speakers": 200,
    "speech_encoder":"vec256l9"
  }
```

### 🆕 Shallow diffusion
![Diagram](docs/assets/shadowdiffusion.png)

## 💬 Python Version

Based on our current validation, the stable baseline for this customized repository is `Python 3.11.x`.

## 🧪 Environment Convention

This project now standardizes on Python's built-in `venv`:

- the virtual environment should be created at the project root as `.venv311`
- Conda / Anaconda are no longer the default environment workflow
- Windows batch launchers are expected to use `.venv311\\Scripts\\python.exe`

Standard initialization:

```shell
python -m venv .venv311
```

## 🚀 Current Recommended Workflow

If you are using this repository in its current customized form, use the app-based workflow first instead of starting from the low-level scripts.

### 1. Create the virtual environment and install dependencies

```shell
python -m venv .venv311
```

```shell
# Windows
.venv311\Scripts\activate

# macOS / Linux
source .venv311/bin/activate

pip install --upgrade pip setuptools wheel
# Windows + NVIDIA GPU (recommended)
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Notes:

- `Windows + NVIDIA GPU` is the primary validated platform right now
- `Linux + NVIDIA GPU` should use the same CUDA-enabled PyTorch approach
- `macOS` may use CPU builds of `torch / torchaudio` for app and inference experiments, but should not be treated as a training platform

### 2. Train models

Recommended entrypoint:

```shell
python -m src.app_train
```

On Windows you can also double-click:

- `启动训练界面.bat`

The training page now starts with dependency preparation:

- checking training dependencies and base models
- importing missing files into standard locations
- preparing training dependencies and base models under `model_assets/dependencies/encoders/`, `model_assets/dependencies/vocoders/`, and `model_assets/dependencies/base_models/44k/`

The current training console supports:

- importing `training_data/source/<speaker_dir>/*.wav`
- suggesting safe directory names
- deleting the current speaker directory
- automatically binding the processed output dir to `training_data/processed/44k/<speaker_dir>`
- defaulting the model output name to the current speaker directory name
- one-click execution for resample, config generation, feature extraction, and main model training
- training checks, stage judgement, task status, logs, and warning/error hints

The current recommended workflow is **single-speaker training**:

- one speaker directory per training run
- one speaker should normally use one dedicated model output directory under `model_assets/workspaces/<model_name>/`
- when you want to train a second speaker, switch to that speaker directory and use a new model output name instead of mixing speakers into the same model

The main top-level directories can now be understood as three groups:

- `training_data`
  - `training_data/source/`: raw training audio
  - `training_data/processed/`: preprocessed training data
- `inference_data`
  - `inference_data/inputs/`: inference inputs
  - `inference_data/outputs/`: inference outputs
- `model_assets`
  - `model_assets/dependencies/`: base models, encoders, vocoders, and other dependencies
  - `model_assets/workspaces/`: per-model training workspaces
  - `model_assets/local/`: local inference model directory

### 3. Run inference

Recommended entrypoint:

```shell
python -m src.app_infer
```

The current inference page supports:

- loading the main model, quality-enhancement model, and timbre-enhancement file
- switching quality modes
- exporting a runtime summary
- an offline high-quality singing-conversion workflow

### 4. When to use the legacy sections below

The remaining `dataset / preprocessing / training / inference` sections are still useful if you want to:

- run low-level scripts directly
- understand the underlying training pipeline
- troubleshoot on a Windows + GPU machine via CLI

## 📥 Pre-trained Model Files

#### **Required**

**You need to select one encoder from the list below**

##### **1. If using contentvec as speech encoder(recommended)**

`vec768l12` uses the current Transformers / HF ContentVec route.

- ContentVec HF directory: `model_assets/dependencies/encoders/contentvec_hf/`
  - Required files:
    - `config.json`
    - `model.safetensors`
  - Locked source currently used by this project:
    - `lengyue233/content-vec-best`

If the directory does not exist locally, the project can fall back to downloading from the locked Hugging Face source. For long-term stability, keeping the local directory is still recommended.

##### **2. If hubertsoft is used as the speech encoder**
- soft vc hubert: [hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)
  - Place it under the `model_assets/dependencies` directory

##### **3. If whisper-ppg as the encoder**
- download model at [medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), the model fits `whisper-ppg`
- or download model at [large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt), the model fits `whisper-ppg-large`
  - Place it under the `model_assets/dependencies` directory
  
##### **4. If cnhubertlarge as the encoder**
- download model at [chinese-hubert-large-fairseq-ckpt.pt](https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt)
  - Place it under the `model_assets/dependencies` directory

##### **5. If dphubert as the encoder**
- download model at [DPHuBERT-sp0.75.pth](https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth)
  - Place it under the `model_assets/dependencies` directory

##### **6. If WavLM is used as the encoder**
- download model at  [WavLM-Base+.pt](https://valle.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D), the model fits `wavlmbase+`
  - Place it under the `model_assets/dependencies` directory

##### **7. If OnnxHubert/ContentVec as the encoder**
- download model at [MoeSS-SUBModel](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main)
  - Place it under the `model_assets/dependencies` directory

#### **List of Encoders**
- "vec768l12"
- "vec256l9"
- "vec256l9-onnx"
- "vec256l12-onnx"
- "vec768l9-onnx"
- "vec768l12-onnx"
- "hubertsoft-onnx"
- "hubertsoft"
- "whisper-ppg"
- "cnhubertlarge"
- "dphubert"
- "whisper-ppg-large"
- "wavlmbase+"

#### **Optional(Strongly recommend)**

- Pre-trained base model files:
  - [G_0.pth (vec768l12)](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/vec768l12/G_0.pth)
  - [D_0.pth (vec768l12)](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/vec768l12/D_0.pth)
  - Place them under the `model_assets/dependencies/base_models/44k` directory

- Diffusion pre-trained base model file:
  - [model_0.pt (diffusion/768l12)](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/diffusion/768l12/model_0.pt)
  - Put it in the `model_assets/dependencies/base_models/44k/diffusion` directory

At training start, the app will sync these base models into the runtime experiment directory `model_assets/workspaces/44k/`. `model_assets/workspaces/44k` is now mainly the output directory for training artifacts, not the long-term storage location for base models.

These links match the current repository's default `vec768l12` / ContentVec route. Do not mix them with same-named files from other encoder folders such as `vec256l9`.

While the pretrained model typically does not pose copyright concerns, it is essential to remain vigilant. It is advisable to consult with the author beforehand or carefully review the description to ascertain the permissible usage of the model. This helps ensure compliance with any specified guidelines or restrictions regarding its utilization.

#### **Optional(Select as Required)**

##### NSF-HIFIGAN

If you are using the `NSF-HIFIGAN enhancer` or `shallow diffusion`, you will need to download the pre-trained NSF-HIFIGAN model.

- Pre-trained NSF-HIFIGAN Vocoder: [nsf_hifigan_20221211.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip)
  - Unzip and place the four files under the `model_assets/dependencies/vocoders/nsf_hifigan` directory

```shell
# nsf_hifigan
wget -P model_assets/dependencies/ https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip
unzip -od model_assets/dependencies/vocoders/nsf_hifigan model_assets/dependencies/nsf_hifigan_20221211.zip
# Alternatively, you can manually download and place it in the model_assets/dependencies/vocoders/nsf_hifigan directory
# URL: https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1
```

##### RMVPE

If you are using the `rmvpe` F0 Predictor, you will need to download the pre-trained RMVPE model.

+ download model at [rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip), this weight is recommended.
  + unzip `rmvpe.zip`，and rename the `model.pt` file to `rmvpe.pt` and place it under the `model_assets/dependencies/encoders` directory.

- ~~download model at [rmvpe.pt](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/rmvpe.pt)~~
  - ~~Place it under the `model_assets/dependencies/encoders` directory~~

##### FCPE(Preview version)

[FCPE(Fast Context-base Pitch Estimator)](https://github.com/CNChTu/MelPE) is a dedicated F0 predictor designed for real-time voice conversion and will become the preferred F0 predictor for sovits real-time voice conversion in the future.(The paper is being written)

If you are using the `fcpe` F0 Predictor, you will need to download the pre-trained FCPE model.

- download model at [fcpe.pt](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt)
  - Place it under the `model_assets/dependencies` directory

## 📊 Dataset Preparation (Low-level Script Reference)

For the current customized repository, it is recommended to import speaker folders from the training page started with `python -m src.app_train`.  
If you use the low-level CLI flow, the dataset structure is still the following:

```
training_data/source
└───your_speaker
    ├───xxx1-xxx1.wav
    ├───...
    └───Lxx-0xx8.wav
```
There are no specific restrictions on the format of the name for each audio file (naming conventions such as `000001.wav` to `999999.wav` are also valid), but the file type must be `WAV``.

You can customize the speaker's name as showed below:

```
training_data/source
└───suijiSUI
    ├───1.wav
    ├───...
    └───25788785-20221210-200143-856_01_(Vocals)_0_0.wav
```

## 🛠️ Preprocessing (Low-level Script Reference)

### 0. Slice audio

To avoid video memory overflow during training or pre-processing, it is recommended to limit the length of audio clips. Cutting the audio to a length of "5s - 15s" is more recommended. Slightly longer times are acceptable, however, excessively long clips may cause problems such as `torch.cuda.OutOfMemoryError`.

To facilitate the slicing process, you can use [audio-slicer-GUI](https://github.com/flutydeer/audio-slicer) or [audio-slicer-CLI](https://github.com/openvpi/audio-slicer)

In general, only the `Minimum Interval` needs to be adjusted. For spoken audio, the default value usually suffices, while for singing audio, it can be adjusted to around `100` or even `50`, depending on the specific requirements.

After slicing, it is recommended to remove any audio clips that are excessively long or too short.

**If you are using whisper-ppg encoder for training, the audio clips must shorter than 30s.**

### 1. Resample to 44100Hz and mono

```shell
python -m src.train_pipeline.resample
```

#### Cautions

Current `train_pipeline/resample.py` only trims silence and resamples audio. Loudness, gain staging, and limiter processing are no longer handled inside the project. Please finish those steps in your own audio workflow before training.

### 2. Automatically split the dataset into training and validation sets, and generate configuration files.

```shell
python -m src.train_pipeline.preprocess_flist_config --speech_encoder vec768l12
```

speech_encoder has the following options

```
vec768l12
vec256l9
hubertsoft
whisper-ppg
cnhubertlarge
dphubert
whisper-ppg-large
wavlmbase+
```

If the speech_encoder argument is omitted, the default value is `vec768l12`

**Use loudness embedding**

Add `--vol_aug` if you want to enable loudness embedding:

```shell
python -m src.train_pipeline.preprocess_flist_config --speech_encoder vec768l12 --vol_aug
```

After enabling loudness embedding, the trained model will match the loudness of the input source; otherwise, it will match the loudness of the training set.

#### You can modify some parameters in the generated config.json and diffusion.yaml

* `keep_ckpts`: Keep the the the number of previous models during training. Set to `0` to keep them all. Default is `3`.

* `all_in_mem`: Load all dataset to RAM. It can be enabled when the disk IO of some platforms is too low and the system memory is **much larger** than your dataset.
  
* `batch_size`: The amount of data loaded to the GPU for a single training session can be adjusted to a size lower than the GPU memory capacity.

* `vocoder_name`: Select a vocoder. The default is `nsf-hifigan`.

##### diffusion.yaml

* `cache_all_data`: Load all dataset to RAM. It can be enabled when the disk IO of some platforms is too low and the system memory is **much larger** than your dataset.

* `duration`: The duration of the audio slicing during training, can be adjusted according to the size of the video memory, **Note: this value must be less than the minimum time of the audio in the training set!**

* `batch_size`: The amount of data loaded to the GPU for a single training session can be adjusted to a size lower than the video memory capacity.

* `timesteps`: The total number of steps in the diffusion model, which defaults to 1000.

* `k_step_max`: Training can only train `k_step_max` step diffusion to save training time, note that the value must be less than `timesteps`, 0 is to train the entire diffusion model, **Note: if you do not train the entire diffusion model will not be able to use only_diffusion!**

##### **List of Vocoders**

```
nsf-hifigan
nsf-snake-hifigan
```

### 3. Generate hubert and f0

```shell
python -m src.train_pipeline.preprocess_hubert_f0 --f0_predictor dio
```

f0_predictor has the following options

```
crepe
dio
pm
harvest
rmvpe
fcpe
```

If the training set is too noisy,it is recommended to use `crepe` to handle f0

If the f0_predictor parameter is omitted, the default value is `rmvpe`

If you want shallow diffusion (optional), you need to add the `--use_diff` parameter, for example:

```shell
python -m src.train_pipeline.preprocess_hubert_f0 --f0_predictor dio --use_diff
```

**Speed Up preprocess**

If your dataset is pretty large,you can increase the param `--num_processes` like that:

```shell
python -m src.train_pipeline.preprocess_hubert_f0 --f0_predictor dio --num_processes 8
```
All the worker will be assigned to different GPU if you have more than one GPUs.

After completing the above steps, the processed dataset directory will contain the preprocessed data, and the training_data/source folder can be deleted.

## 🏋️‍ Training (Low-level Script Reference)

### Sovits Model

```shell
python -m src.train_pipeline.train -c model_assets/workspaces/<model_name>/config.json -m <model_name>
```

### Diffusion Model (optional)

If the shallow diffusion function is needed, the diffusion model needs to be trained. The diffusion model training method is as follows:

```shell
python -m src.train_pipeline.train_diff -c model_assets/workspaces/<model_name>/diffusion.yaml
```

During training, the model files will be saved to `model_assets/workspaces/<model_name>/`, and the diffusion model will be saved to `model_assets/workspaces/<model_name>/diffusion/`.

## 🤖 Inference (Low-level Script Reference)

If you are using the current customized repository, prefer `python -m src.app_infer`.  
The section below is the CLI reference for `python -m src.services.inference_main`.

```shell
# Example
python -m src.services.inference_main -m "model_assets/workspaces/<model_name>/G_30400.pth" -c "model_assets/workspaces/<model_name>/config.json" -n "君の知らない物語-src.wav" -t 0 -s "<speaker_name>"
```

Required parameters:
- `-m` | `--model_path`: path to the model.
- `-c` | `--config_path`: path to the configuration file.
- `-n` | `--clean_names`: a list of wav file names located in the `inference_data/inputs` folder.
- `-t` | `--trans`: pitch shift, supports positive and negative (semitone) values.
- `-s` | `--speaker`: the speaker name used by the current model.
- `-cl` | `--clip`: Forced audio clipping, set to 0 to disable(default), setting it to a non-zero value (duration in seconds) to enable.

Optional parameters: see the next section
- `-lg` | `--linear_gradient`: The cross fade length of two audio slices in seconds. If there is a discontinuous voice after forced slicing, you can adjust this value. Otherwise, it is recommended to use the default value of 0.
- `-f0p` | `--f0_predictor`: Select a F0 predictor, options are `crepe`, `pm`, `dio`, `harvest`, `rmvpe`,`fcpe`, default value is `pm`(note: f0 mean pooling will be enable when using `crepe`)
- `-a` | `--auto_predict_f0`: automatic pitch prediction, do not enable this when converting singing voices as it can cause serious pitch issues.
- `-cm` | `--cluster_model_path`: Cluster model or feature retrieval index path, if left blank, it will be automatically set as the default path of these models. If there is no training cluster or feature retrieval, fill in at will.
- `-cr` | `--cluster_infer_ratio`: The proportion of clustering scheme or feature retrieval ranges from 0 to 1. If there is no training clustering model or feature retrieval, the default is 0.
- `-eh` | `--enhance`: Whether to use NSF_HIFIGAN enhancer, this option has certain effect on sound quality enhancement for some models with few training sets, but has negative effect on well-trained models, so it is disabled by default.
- `-shd` | `--shallow_diffusion`: Whether to use shallow diffusion, which can solve some electrical sound problems after use. This option is disabled by default. When this option is enabled, NSF_HIFIGAN enhancer will be disabled
- `-lea` | `--loudness_envelope_adjustment`：The adjustment of the input source's loudness envelope in relation to the fusion ratio of the output loudness envelope. The closer to 1, the more the output loudness envelope is used
- `-fr` | `--feature_retrieval`：Whether to use feature retrieval If clustering model is used, it will be disabled, and `cm` and `cr` parameters will become the index path and mixing ratio of feature retrieval
  
Shallow diffusion settings:
- `-dm` | `--diffusion_model_path`: Diffusion model path
- `-dc` | `--diffusion_config_path`: Diffusion config file path
- `-ks` | `--k_step`: The larger the number of k_steps, the closer it is to the result of the diffusion model. The default is 100
- `-od` | `--only_diffusion`: Whether to use Only diffusion mode, which does not load the sovits model to only use diffusion model inference
- `-se` | `--second_encoding`：which involves applying an additional encoding to the original audio before shallow diffusion. This option can yield varying results - sometimes positive and sometimes negative.

### Cautions

If inferencing using `whisper-ppg` speech encoder, you need to set `--clip` to 25 and `-lg` to 1. Otherwise it will fail to infer properly.

## 🤔 Optional Settings

If you are satisfied with the previous results, or if you do not feel you understand what follows, you can skip it and it will have no effect on the use of the model. The impact of these optional settings mentioned is relatively small, and while they may have some impact on specific datasets, in most cases the difference may not be significant.

### Automatic f0 prediction

During the training of the 4.0 model, an f0 predictor is also trained, which enables automatic pitch prediction during voice conversion. However, if the results are not satisfactory, manual pitch prediction can be used instead. Please note that when converting singing voices, it is advised not to enable this feature as it may cause significant pitch shifting.

- Set `auto_predict_f0` to `true` in `services/inference_main.py`.

### Cluster-based timbre leakage control

Introduction: The clustering scheme implemented in this model aims to reduce timbre leakage and enhance the similarity of the trained model to the target's timbre, although the effect may not be very pronounced. However, relying solely on clustering can reduce the model's clarity and make it sound less distinct. Therefore, a fusion method is adopted in this model to control the balance between the clustering and non-clustering approaches. This allows manual adjustment of the trade-off between "sounding like the target's timbre" and "have clear enunciation" to find an optimal balance.

No changes are required in the existing steps. Simply train an additional clustering model, which incurs relatively low training costs.

- Training process:
  - Train on a machine with good CPU performance. According to extant experience, it takes about 4 minutes to train each speaker on a Tencent Cloud machine with 6-core CPU.
  - Execute `python cluster/train_cluster.py`. The output model will be saved in `model_assets/workspaces/44k/kmeans_10000.pt`.
  - The clustering model can currently be trained using the gpu by executing `python cluster/train_cluster.py --gpu`
- Inference process:
  - Specify `cluster_model_path` in `services/inference_main.py`. If not specified, the default is `model_assets/workspaces/44k/kmeans_10000.pt`.
  - Specify `cluster_infer_ratio` in `services/inference_main.py`, where `0` means not using clustering at all, `1` means only using clustering, and usually `0.5` is sufficient.

### Feature retrieval

Introduction: As with the clustering scheme, the timbre leakage can be reduced, the enunciation is slightly better than clustering, but it will reduce the inference speed. By employing the fusion method, it becomes possible to linearly control the balance between feature retrieval and non-feature retrieval, allowing for fine-tuning of the desired proportion.

- Training process: 
  First, it needs to be executed after generating hubert and f0: 

```shell
python -m src.train_pipeline.train_index -c model_assets/workspaces/<model_name>/config.json
```

The output of the model will be in `model_assets/workspaces/<model_name>/feature_and_index.pkl`

- Inference process: 
  - The `--feature_retrieval` needs to be formulated first, and the clustering mode automatically switches to the feature retrieval mode.
  - Specify `cluster_model_path` in `services/inference_main.py`. If not specified, the default is `model_assets/workspaces/44k/feature_and_index.pkl`.
  - Specify `cluster_infer_ratio` in `services/inference_main.py`, where `0` means not using feature retrieval at all, `1` means only using feature retrieval, and usually `0.5` is sufficient.

## 🗜️ Model compression

The generated model contains data that is needed for further training. If you confirm that the model is final and not be used in further training, it is safe to remove these data to get smaller file size (about 1/3).

```shell
# Example
python -m src.tools.compress_model -c="model_assets/workspaces/<model_name>/config.json" -i="model_assets/workspaces/<model_name>/G_30400.pth" -o="model_assets/workspaces/<model_name>/release.pth"
```

## 📤 Exporting to Onnx

Use [tools/onnx_export.py](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/onnx_export.py)

- Create a folder named `checkpoints` and open it
- Create a folder in the `checkpoints` folder as your project folder, naming it after your project, for example `aziplayer`
- Rename your model as `model.pth`, the configuration file as `config.json`, and place them in the `aziplayer` folder you just created
- Modify `"NyaruTaffy"` in `path = "NyaruTaffy"` in [tools/onnx_export.py](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/onnx_export.py) to your project name, for example `path = "aziplayer"`
- Run `python -m src.tools.onnx_export`
- Wait for it to finish running. A `model.onnx` will be generated in your project folder, which is the exported model.

Note: For Hubert Onnx models, please use the models provided by MoeSS. Currently, they cannot be exported on their own (Hubert in fairseq has many unsupported operators and things involving constants that can cause errors or result in problems with the input/output shape and results when exported.)


## 📎 Reference

| URL | Designation | Title | Implementation Source |
| --- | ----------- | ----- | --------------------- |
|[2106.06103](https://arxiv.org/abs/2106.06103) | VITS (Synthesizer)| Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech | [jaywalnut310/vits](https://github.com/jaywalnut310/vits) |
|[2111.02392](https://arxiv.org/abs/2111.02392) | SoftVC (Speech Encoder)| A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion | [bshall/hubert](https://github.com/bshall/hubert) |
|[2204.09224](https://arxiv.org/abs/2204.09224) | ContentVec (Speech Encoder)| ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers | [auspicious3000/contentvec](https://github.com/auspicious3000/contentvec) |
|[2212.04356](https://arxiv.org/abs/2212.04356) | Whisper (Speech Encoder) | Robust Speech Recognition via Large-Scale Weak Supervision | [openai/whisper](https://github.com/openai/whisper) |
|[2110.13900](https://arxiv.org/abs/2110.13900) | WavLM (Speech Encoder) | WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing | [microsoft/unilm/wavlm](https://github.com/microsoft/unilm/tree/master/wavlm) |
|[2305.17651](https://arxiv.org/abs/2305.17651) | DPHubert (Speech Encoder) | DPHuBERT: Joint Distillation and Pruning of Self-Supervised Speech Models | [pyf98/DPHuBERT](https://github.com/pyf98/DPHuBERT) |
|[DOI:10.21437/Interspeech.2017-68](http://dx.doi.org/10.21437/Interspeech.2017-68) | Harvest (F0 Predictor) | Harvest: A high-performance fundamental frequency estimator from speech signals | [mmorise/World/harvest](https://github.com/mmorise/World/blob/master/src/harvest.cpp) |
|[aes35-000039](https://www.aes.org/e-lib/online/browse.cfm?elib=15165) | Dio (F0 Predictor) | Fast and reliable F0 estimation method based on the period extraction of vocal fold vibration of singing voice and speech | [mmorise/World/dio](https://github.com/mmorise/World/blob/master/src/dio.cpp) |
|[8461329](https://ieeexplore.ieee.org/document/8461329) | Crepe (F0 Predictor) | Crepe: A Convolutional Representation for Pitch Estimation | [maxrmorrison/torchcrepe](https://github.com/maxrmorrison/torchcrepe) |
|[DOI:10.1016/j.wocn.2018.07.001](https://doi.org/10.1016/j.wocn.2018.07.001) | Parselmouth (F0 Predictor) | Introducing Parselmouth: A Python interface to Praat | [YannickJadoul/Parselmouth](https://github.com/YannickJadoul/Parselmouth) |
|[2306.15412v2](https://arxiv.org/abs/2306.15412v2) | RMVPE (F0 Predictor) | RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music | [Dream-High/RMVPE](https://github.com/Dream-High/RMVPE) |
|[2010.05646](https://arxiv.org/abs/2010.05646) | HIFIGAN (Vocoder) | HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | [jik876/hifi-gan](https://github.com/jik876/hifi-gan) |
|[1810.11946](https://arxiv.org/abs/1810.11946.pdf) | NSF (Vocoder) | Neural source-filter-based waveform model for statistical parametric speech synthesis | [openvpi/DiffSinger/modules/nsf_hifigan](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan)
|[2006.08195](https://arxiv.org/abs/2006.08195) | Snake (Vocoder) | Neural Networks Fail to Learn Periodic Functions and How to Fix It | [EdwardDixon/snake](https://github.com/EdwardDixon/snake)
|[2105.02446v3](https://arxiv.org/abs/2105.02446v3) | Shallow Diffusion (PostProcessing)| DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism | [CNChTu/Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) |
|[K-means](https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=01D65490BADCC216F350D06F84D721AD?doi=10.1.1.308.8619&rep=rep1&type=pdf) | Feature K-means Clustering (PreProcessing)| Some methods for classification and analysis of multivariate observations | This repo |
| | Feature TopK Retrieval (PreProcessing)| Retrieval based Voice Conversion | [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) |
| | whisper ppg| whisper ppg | [PlayVoice/whisper_ppg](https://github.com/PlayVoice/whisper_ppg) |
| | bigvgan| bigvgan | [PlayVoice/so-vits-svc-5.0](https://github.com/PlayVoice/so-vits-svc-5.0/tree/bigvgan-mix-v2/vits_decoder/alias) |


## ☀️ Previous contributors

For some reason the author deleted the original repository. Because of the negligence of the organization members, the contributor list was cleared because all files were directly reuploaded to this repository at the beginning of the reconstruction of this repository. Now add a previous contributor list to README.md.

*Some members have not listed according to their personal wishes.*

<table>
  <tr>
    <td align="center"><a href="https://github.com/MistEO"><img src="https://avatars.githubusercontent.com/u/18511905?v=4" width="100px;" alt=""/><br /><sub><b>MistEO</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/XiaoMiku01"><img src="https://avatars.githubusercontent.com/u/54094119?v=4" width="100px;" alt=""/><br /><sub><b>XiaoMiku01</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/ForsakenRei"><img src="https://avatars.githubusercontent.com/u/23041178?v=4" width="100px;" alt=""/><br /><sub><b>しぐれ</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/TomoGaSukunai"><img src="https://avatars.githubusercontent.com/u/25863522?v=4" width="100px;" alt=""/><br /><sub><b>TomoGaSukunai</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/Plachtaa"><img src="https://avatars.githubusercontent.com/u/112609742?v=4" width="100px;" alt=""/><br /><sub><b>Plachtaa</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/zdxiaoda"><img src="https://avatars.githubusercontent.com/u/45501959?v=4" width="100px;" alt=""/><br /><sub><b>zd小达</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/Archivoice"><img src="https://avatars.githubusercontent.com/u/107520869?v=4" width="100px;" alt=""/><br /><sub><b>凍聲響世</b></sub></a><br /></td>
  </tr>
</table>

## 📚 Some legal provisions for reference

#### Any country, region, organization, or individual using this project must comply with the following laws.

#### 《民法典》

##### 第一千零一十九条 

任何组织或者个人不得以丑化、污损，或者利用信息技术手段伪造等方式侵害他人的肖像权。未经肖像权人同意，不得制作、使用、公开肖像权人的肖像，但是法律另有规定的除外。未经肖像权人同意，肖像作品权利人不得以发表、复制、发行、出租、展览等方式使用或者公开肖像权人的肖像。对自然人声音的保护，参照适用肖像权保护的有关规定。

#####  第一千零二十四条 

【名誉权】民事主体享有名誉权。任何组织或者个人不得以侮辱、诽谤等方式侵害他人的名誉权。  

#####  第一千零二十七条

【作品侵害名誉权】行为人发表的文学、艺术作品以真人真事或者特定人为描述对象，含有侮辱、诽谤内容，侵害他人名誉权的，受害人有权依法请求该行为人承担民事责任。行为人发表的文学、艺术作品不以特定人为描述对象，仅其中的情节与该特定人的情况相似的，不承担民事责任。  

#### 《[中华人民共和国宪法](http://www.gov.cn/guoqing/2018-03/22/content_5276318.htm)》

#### 《[中华人民共和国刑法](http://gongbao.court.gov.cn/Details/f8e30d0689b23f57bfc782d21035c3.html?sw=%E4%B8%AD%E5%8D%8E%E4%BA%BA%E6%B0%91%E5%85%B1%E5%92%8C%E5%9B%BD%E5%88%91%E6%B3%95)》

#### 《[中华人民共和国民法典](http://gongbao.court.gov.cn/Details/51eb6750b8361f79be8f90d09bc202.html)》

#### 《[中华人民共和国合同法](http://www.npc.gov.cn/zgrdw/npc/lfzt/rlyw/2016-07/01/content_1992739.htm)》

## 💪 Thanks to all contributors for their efforts
<a href="https://github.com/svc-develop-team/so-vits-svc/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=svc-develop-team/so-vits-svc" />
</a>
