# Relation-R1

## 🛠️ Installation
To install using pip:
```shell
pip install ms-swift -U
```

To install from source:
```shell
# pip install git+https://github.com/modelscope/ms-swift.git

git clone git@github.com:HKUST-LongGroup/Relation-R1.git #3.3 version
pip install -e .
```

```shell
bash requirements/install_all.sh
```

Running Environment:

|              | Range                | Recommended | Notes                                     |
| ------------ | -------------------- | ----------- | ----------------------------------------- |
| python       | >=3.9                | 3.10        |                                           |
| cuda         |                      | cuda12      | No need to install if using CPU, NPU, MPS |
| torch        | >=2.0                |             |                                           |
| transformers | >=4.33               | 4.49      |                                           |
| modelscope   | >=1.19               |             |                                           |
| peft | >=0.11,<0.15 | ||
| trl | >=0.13,<0.17 | 0.15 |RLHF|
| deepspeed    | >=0.14 | 0.14.5 | Training                                  |
| vllm         | >=0.5.1              | 0.7.3       | Inference/Deployment/Evaluation           |
| lmdeploy     | lmdeploy>=0.5 | 0.7.1       | Inference/Deployment/Evaluation           |
| evalscope | >=0.11 |  | Evaluation |

For more optional dependencies, you can refer to [here](https://github.com/modelscope/ms-swift/blob/main/requirements/install_all.sh).


## 🚀 Quick Start

### Command Train
Execute the training code for all stages according to the order of the shell scripts in the useful/train directory.
For example:

```shell
bash useful_sh/train/1.fix_sft.sh
```

### Command SGG/GSR Infer

```shell
bash useful_sh/infer/sgg_infer.sh
```

### Command SGG/GSR Eval

```shell
bash useful_sh/eval/sgg_eval.sh
```

## 🗓️ TODO
- [x] Release initial code  
- [ ] Release training/test data
- [ ] Release Relation-R1 checkpoint


## 🖊️ BibTeX
If you find this project useful in your research, please consider cite:

```bibtex
@inproceedings{Lin2025relr1,
title={Relation-R1: Progressively Cognitive Chain-of-Thought Guided Reinforcement Learning for Unified Relation Comprehension},
booktitle={AAAI Conference on Artificial Intelligence},
author={Lin Li and Wei Chen and Jiahui Li and Kwang-Ting Cheng and Long Chen},
date={2026},
}
```

