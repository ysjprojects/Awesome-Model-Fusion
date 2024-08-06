# Awesome Large Language Model (LLM) Merging
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)  ![visitor badge](https://visitor-badge.lithub.cc/badge?page_id=ysjprojects.Awesome-LLM-Merging&left_text=Visitors) ![GitHub stars](https://img.shields.io/github/stars/ysjprojects/Awesome-LLM-Merging?color=yellow) ![GitHub forks](https://img.shields.io/github/forks/ysjprojects/Awesome-LLM-Merging?color=9cf) [![GitHub license](https://img.shields.io/github/license/ysjprojects/Awesome-LLM-Merging)](https://github.com/ysjprojects/Awesome-LLM-Merging/blob/main/LICENSE)

[Merge-chan says hi!](assets/merge-chan.png)

This is a collection of research papers for **Large Language Model (LLM) Merging**.
And the repository will be continuously updated to track the frontier of LLM Merging.

Welcome to follow and star!


## Table of Contents

- [Awesome Large Language Model (LLM) Merging](#awesome-large-language-model-llm-merging)
  - [Table of Contents](#table-of-contents)
  - [Papers](#papers)
    - [2024](#2024)
    - [2023](#2023)
    - [2022 and before](#2022-and-before)
  - [Codebases](#codebases)
  - [Blogs](#blogs)
  - [Contributing](#contributing)
  - [License](#license)

## Papers

```
format:
- [title](paper link) [links]
  - author1, author2, and author3...
  - publisher
  - keyword
  - code
  - experiment environments and datasets
```

### 2024
- [Checkpoint Merging via Bayesian Optimization in LLM Pretraining](https://arxiv.org/pdf/2403.19390)
  - Deyuan Liu, Zecheng Wang, Bingning Wang, Weipeng Chen, Chunshan Li, Zhiying Tu, Dianhui Chu, Bo Li, Dianbo Sui
  - Keyword: Pretraining, Checkpoint Merging, Bayesian Optimization

- [Arcee's MergeKit: A Toolkit for Merging Large Language Models](https://arxiv.org/pdf/2403.13257v2)
  - Charles Goddard, Shamane Siriwardhana, Malikeh Ehghaghi, Luke Meyers, Vlad Karpukhin, Brian Benedict, Mark McQuade, Jacob Solawetz
  - Keyword: Model Merging, Task Arithmetic
  - Code: [Official](https://github.com/arcee-ai/mergekit)

- [WARM: On the Benefits of Weight Averaged Reward Models](https://arxiv.org/pdf/2401.12187)
  - Alexandre Ramé, Nino Vieillard, Léonard Hussenot, Robert Dadashi, Geoffrey Cideron, Olivier Bachem, Johan Ferret
  - Keyword: RLHF, Weight Averaged Reward Models

- [Model Stock: All we need is just a few fine-tuned models](https://arxiv.org/pdf/2403.19522)
  - Dong-Hwan Jang, Sangdoo Yun, Dongyoon Han
  - Keyword: Layer-wise Weight Averaging
  - Code: [Official](https://github.com/naver-ai/model-stock)

- [Evolutionary Optimization of Model Merging Recipes](https://arxiv.org/pdf/2403.13187v1)
  - Takuya Akiba, Makoto Shing, Yujin Tang, Qi Sun, David Ha
  - Keyword: Evolutionary Algorithms, Cross-Domain Merging, Automated Model Composition
  - Code: [Official](https://github.com/sakanaai/evolutionary-model-merge)

- [Training-Free Pretrained Model Merging](https://arxiv.org/pdf/2403.01753v3)
  - Zhengqi Xu, Ke Yuan, Huiqiong Wang, Yong Wang, Mingli Song, Jie Song
  - Keyword: Training-free Model Merging
  - Code: [Official](https://github.com/zju-vipa/training_free_model_merging)

- [DELLA-Merging: Reducing Interference in Model Merging through Magnitude-Based Sampling](https://arxiv.org/pdf/2406.11617v1)
  - Pala Tej Deep, Rishabh Bhardwaj, Soujanya Poria
  - Keyword: MAGPRUNE, Pruning Technique, TIES, DARE
  - Code: [Official](https://github.com/declare-lab/della)
  
- [Merging Multi-Task Models via Weight-Ensembling Mixture of Experts](https://arxiv.org/pdf/2402.00433v2)
  - Anke Tang, Li Shen, Yong Luo, Nan Yin, Lefei Zhang, Dacheng Tao
  - Keyword: Multi-Task Learning, Task Arithmetic, Vision Transformers, Weight Ensembling, Mixture Of Experts
  - Code: [Official](https://github.com/tanganke/weight-ensembling_moe)
 
- [Twin-Merging: Dynamic Integration of Modular Expertise in Model Merging](https://arxiv.org/abs/2406.15479v1)
  - Zhenyi Lu, Chenghao Fan, Wei Wei, Xiaoye Qu, Dangyang Chen, Yu Cheng
  - Keyword: Knowledge Modularization, Dynamic Merging
  - Code: [Official](https://github.com/LZY-the-boys/Twin-Merging)

- [Domain Adaptation of Llama3-70B-Instruct through Continual Pre-Training and Model Merging: A Comprehensive Evaluation](https://arxiv.org/pdf/2406.14971)
  - Shamane Siriwardhana, Mark McQuade, Thomas Gauthier, Lucas Atkins, Fernando Fernandes Neto, Luke Meyers, Anneketh Vij, Tyler Odenthal, Charles Goddard, Mary MacCarthy, Jacob Solawetz
  - Keyword: Domain Adaptation
  - Code: [Official](https://github.com/LZY-the-boys/Twin-Merging)

### 2023
- [LM-Cocktail: Resilient Tuning of Language Models via Model Merging](https://arxiv.org/pdf/2311.13534v4)
  - Shitao Xiao, Zheng Liu, Peitian Zhang, Xingrun Xing
  - Keyword: Model Merging, Catastrophic Forgetting, Resilient Tuning
  - Code: [Official](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)

- [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/pdf/2311.03099)
  - Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, Yongbin Li
  - Keyword: DARE
  - Code: [Official](https://github.com/yule-buaa/mergelm)

- [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/pdf/2306.01708v2)
  - Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, Mohit Bansal
  - Keyword: Transfer Learning, TIES, Parameter Interference
  - Code: [Official](https://github.com/prateeky2806/ties-merging)

- [Personalized Soups: Personalized Large Language Model Alignment via Post-hoc Parameter Merging](https://arxiv.org/pdf/2310.11564v1)
  - Joel Jang, Seungone Kim, Bill Yuchen Lin, Yizhong Wang, Jack Hessel, Luke Zettlemoyer, Hannaneh Hajishirzi, Yejin Choi, Prithviraj Ammanabrolu
  - Keyword: LLM Alignment, Post-hoc Parameter Merging, Multi-Objective Reinforcement Learning, RLHF
  - Code: [Official](https://github.com/joeljang/RLPHF)

- [ZipIt! Merging Models from Different Tasks without Training](https://arxiv.org/pdf/2305.03053v3)
  - George Stoica, Daniel Bolya, Jakob Bjorner, Pratik Ramesh, Taylor Hearn, Judy Hoffman
  - Keyword: Multi-task Model Merging, without training
  - Code: [Official](https://github.com/gstoica27/zipit)

- [An Empirical Study of Multimodal Model Merging](https://arxiv.org/pdf/2304.14933v2)
  - Yi-Lin Sung, Linjie Li, Kevin Lin, Zhe Gan, Mohit Bansal, Lijuan Wang
  - Keyword: Multimodal Model Merging
  - Code: [Official](https://github.com/ylsung/vl-merging)

- [AdaMerging: Adaptive Model Merging for Multi-Task Learning](https://arxiv.org/pdf/2310.02575v2)
  - Enneng Yang, Zhenyi Wang, Li Shen, Shiwei Liu, Guibing Guo, Xingwei Wang, Dacheng Tao
  - Keyword: AdaMerging, Multi-Task Learning
  - Code: [Official](https://github.com/EnnengYang/AdaMerging)

- [Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models](https://arxiv.org/pdf/2305.12827v3)
  - Guillermo Ortiz-Jimenez, Alessandro Favero, Pascal Frossard
  - Keyword: Task Arithmetic, VLLM
  - Code: [Official](https://github.com/gortizji/tangent_task_arithmetic)

- [Composing Parameter-Efficient Modules with Arithmetic Operations](https://arxiv.org/pdf/2306.14870)
  - Jinghan Zhang, Shiqi Chen, Junteng Liu, Junxian He
  - Keyword: Task Arithmetic, Parameter-Efficient Modules, LoRa, IA3, Unlearning, Domain Transfer
  - Code: [Official](https://github.com/hkust-nlp/PEM_composition)

### 2022 and before

- [Dataless Knowledge Fusion by Merging Weights of Language Models](https://arxiv.org/pdf/2212.09849v5)
  - Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, Pengxiang Cheng
  - Keyword: Knowledge Fusion, Regression Mean (RegMean) 
  - Code: [Official](https://github.com/bloomberg/dataless-model-merging)

- [Editing Models with Task Arithmetic](https://arxiv.org/pdf/2212.04089v3)
  - Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi
  - Keyword: Task Arithmetic
  - Code: [Official](https://github.com/mlfoundations/task_vectors)

- [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/pdf/2203.05482v3)
  - Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, Ludwig Schmidt
  - Keyword: Domain Generalization, Unsupervised Domain Adaptation, Image Classification
  - Code: [Official](https://github.com/mlfoundations/model-soups)

- [Merging Models with Fisher-Weighted Averaging](https://arxiv.org/pdf/2111.09832)
  - Michael Matena, Colin Raffel
  - Keyword: Model Merging, Fisher-Weighted Averaging
  - Code: [Official](https://github.com/mmatena/model_merging)

- [MergeDistill: Merging Pre-trained Language Models using Distillation](https://arxiv.org/pdf/2106.02834)
  - Simran Khanuja, Melvin Johnson, Partha Talukdar
  - Keyword: Cross-lingual Transfer, Task-agnostic Knowledge Distillation
  - Code: [Official](https://github.com/mmatena/model_merging)

## Codebases
```
format:
- [title](codebase link) [links]
  - author1, author2, and author3...
  - keyword
  - experiment environments, datasets or tasks
```

- [mergekit](https://github.com/arcee-ai/mergekit)
  - arcee-ai
  - Keyword: toolkit, pre-trained LM merging, GPU or CPU execution
  - Task: A toolkit for merging pre-trained language model (Supports Llama, Mistral, GPT-NeoX, StableLM, and more).

- [MergeLM](https://github.com/yule-BUAA/MergeLM)
  - Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, Yongbin Li
  - Keyword: DARE, research
  - Task: Well-coded implementation of five model merging methods, including Average Merging, Task Arithmetic, Fisher Merging, RegMean, and TIES-Merging, combined with the proposed DARE.

- [evolutionary-model-merge](https://github.com/SakanaAI/evolutionary-model-merge)
  - Takuya Akiba, Makoto Shing, Yujin Tang, Qi Sun, David Ha
  - Keyword: Evolutionary Algorithms, Cross-Domain Merging, Automated Model Composition
  - Task: Codebase for the paper [Evolutionary Optimization of Model Merging Recipes](https://arxiv.org/abs/2403.13187).

## Blogs

- [HuggingFace] [Model merging](https://huggingface.co/docs/peft/en/developer_guides/model_merging)
- [HuggingFace] [Merge Large Language Models with mergekit](https://huggingface.co/blog/mlabonne/merge-models)
- [LightningAI] [Efficient Linear Model Merging for LLMs](https://lightning.ai/lightning-ai/studios/efficient-linear-model-merging-for-llms)
- [YouTube] [Deep dive: model merging](https://www.youtube.com/watch?v=cvOpX75Kz4M)
- [YouTube] [Deep dive: model merging, part 2](https://www.youtube.com/watch?v=qbAvOgGmFuE)

## Contributing

Our purpose is to make this repo even better. If you are interested in contributing, please refer to [HERE](CONTRIBUTING.md) for instructions in contribution.

## License

Awesome LLM Merging is released under the Apache 2.0 license.
