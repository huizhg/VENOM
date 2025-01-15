# VENOM
This is the code implementation for our paper, ***VENOM: Text-driven Unrestricted Adversarial Example Generation with Diffusion Models***
---
## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Citation&Acknowledgment](#citationacknowledgment)
---

## Introduction

VENOM is the first text-driven framework for high-quality unrestricted ad**V**ersarial **E**xamples ge**N**eration through diffusi**O**n **M**odels. VENOM unifies image content generation and adversarial synthesis into a single reverse diffusion process, enabling high-fidelity adversarial examples without sacrificing attack success rate (ASR). To stabilize this process, we incorporate an adaptive adversarial guidance strategy with momentum, ensuring that the generated adversarial examples $x^*$ align with the distribution $p(x)$ of natural images. Extensive experiments demonstrate that VENOM achieves superior ASR and image quality compared to prior methods, marking a significant advancement in adversarial example generation and providing insights into model vulnerabilities for improved defense development.

This repository mainly contains three parts:
1. **demo:** 
 You can easily test and explore the functionality of VENOM. For example, given a description of an image ("A juicy cheeseburger with lettuce, tomato, and pickles on a toasted sesame seed bun, served on a diner table.") and a category ("panda") that you want the image to be classified as, VENOM will generate a cheeseburger image based on your description, but it will be classified as a panda by the victim model.
2. **UAE:** 
 UAE stands for *Unrestricted Adversarial Example*. When you use VENOM in UAE mode, reference images and their corresponding labels are required.
3. **NAE:** 
 NAE stands for *Natural Adversarial Example*. VENOM can produce this type of adversarial example using only a text prompt, without the need for reference images. Refer to the example in the demo section.

## Installation
```bash
# Clone the repository
git clone https://github.com/huizhg/VENOM.git
cd VENOM
# Install dependencies
pip install -r reqirements.txt
```
## Usage
1. **demo:**
There are three predefined prompts: "cheeseburger", "fire truck" and "espresso maker" in `run_venom_demo.py`. You are free to try other prompts by modifying these parameters. Note that the target class you want the victim model to misclassify is limited to the 1000 class of ImageNet. Example of running the demo:
  ```bash
  python run_venom_demo.py --target_text "giant panda"

  ```
2. **UAE:**
UAE mode requires reference images to generate adversarial examples, so you need to specify the path to the dataset and its labels. The `--beta` parameter specifies the momentum coefficient.
```bash
python run_venom_uae.py --images_root "path-to-the-dataset" \ 
--label_path "path-to-the-labels-of-the-dataset" \
--beta 0.5 \
--save_dir "./out/venom_uae" 

```

3. **NAE:**
NAE mode creates adversarial examples from random noise. Use `--test True` to generate NAEs for six default testing classes.
```bash
python run_venom_nae.py --test True
```


## Citation&Acknowledgment
If you find this paper or code useful, please consider citing it.

```bibtex
@misc{venom,
      title={VENOM: Text-driven Unrestricted Adversarial Example Generation with Diffusion Models}, 
      author={Hui Kuurila-Zhang and Haoyu Chen and Guoying Zhao},
      year={2025},
      eprint={2501.07922},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.07922}, 
}
```
Thanks to the open-source projects [DiffAttack](https://github.com/WindVChen/DiffAttack) and [AdvDiff](https://github.com/EricDai0/advdiff). Some of our code is based on them.
