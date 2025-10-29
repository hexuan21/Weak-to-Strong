# Weak-to-Strong
Official repo for the paper ["Guiding Through Complexity: What Makes Good Supervision for Hard Reasoning Tasks?"](https://arxiv.org/abs/2410.20533) (Accepted at NAACL 2025 as Oral)

<a target="_blank" href="https://arxiv.org/abs/2410.20533">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a>

<a target="_blank" href="https://github.com/hexuan21/Weak-to-Strong">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-blue?style=flat&logo=github"></a>

<a target="_blank" href="https://huggingface.co/datasets/hexuan21/weak-to-strong">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-orange?style=flat"></a>

<a target="_blank" href="https://hexuan21.github.io/Weak-to-Strong/">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Wegpage-green?style=flat"></a>

<a target="_blank" href="https://github.com/hexuan21/Weak-to-Strong/blob/main/assets/slides.pdf">
<img style="height:22pt" src="https://img.shields.io/badge/-🔎%20Slides-yellow?style=flat"></a>



<br>

## Introduction
How can “weak teacher models” such as average human annotators or existing AI systems, effectively supervise LLMs to improve performance on hard reasoning tasks, especially those that challenge and requires expertise or daily practice from the teacher models? In this paper, we seek for empirical answers to this question by investigating various data-driven strategies that offer supervision data at different quality levels upon tasks of varying complexity. Two intuitive strategies emerge for teacher models to provide supervision during alignment training: 1) using lower-quality supervision from complete tasks that match the difficulty of the target reasoning tasks, and 2) leveraging higher-quality supervision from easier subtasks that are less challenging. Interestingly, we find that even when the outcome error rate for hard task supervision is high (e.g., 90%), training on such data can outperform perfectly correct supervision on easier subtasks on multiple hard math benchmarks. We further identify a more critical factor influencing training performance: step-wise error rates, which indicate the severity of errors in solutions. Specifically, training on hard task supervision with the same outcome error rates but disparate step-wise error rates can lead to a 30% accuracy gap on MATH benchmark. Our results also reveal that supplementing hard task supervision with the corresponding subtask supervision can yield notable performance improvements than simply combining rephrased hard full task supervision, suggesting new avenues for data augmentation.

## Data
All training and test data are available [here](https://huggingface.co/datasets/hexuan21/weak-to-strong/tree/main).

Dataset components:

(1) [task_sol_pairs](https://huggingface.co/datasets/hexuan21/weak-to-strong/tree/main/task_sol_pairs): 
   `{full-task, sub-task}` pairs used for the main results. We compare supervision on easy sub-tasks vs. hard full tasks and measure the impact on model performance (see **Section 5** of the paper).

(2) [weaker_sampling](https://huggingface.co/datasets/hexuan21/weak-to-strong/tree/main/weaker_sampling): 
   Solutions sampled from a weaker teacher model to study how **error severity** in solutions affects model performance (see **Section 6**).

(3) [paraphrase](https://huggingface.co/datasets/hexuan21/weak-to-strong/tree/main/paraphrase):
   Paraphrased task–solution data to explore combining **low-quality, hard-task supervision** (more annotation/sampling errors) with **high-quality, easy-task supervision** (fewer errors) (see **Section 7**).

Before training, convert each data entry to the format required by your codebase (e.g., `sharegpt`, `alpaca`, etc.).

The data construction pipeline (decomposition, answer sampling, checking & filtering, paraphrasing) is located under the `pipeline/` directory and is being updated.


## Training
We use codebase of [alignment-handbook](https://github.com/huggingface/alignment-handbook) to do Supervied Fine-Tuning (SFT), but it is recommended to try a new codebase [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), supporting 100+ LLMs traning.

We take [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) as base-model, and perform a 2-stage fine-tuning. 

(1) Firstly, We use a 164K dataset with elementary school and middle school level math tasks (from the OrcaMath subset of NuminaMath-CoT, and MATH level 1-3) for fine-tuning to establish a base model equipped with foundational instruction-following and mathematical reasoning capabilities.

(2) Then, we fine-tune the model with our synthesized easy sub task and hard full task supervision datasets.

Hyper-params config: 
```
bf16: true
gradient_accumulation_steps: 8
learning_rate: 2.0e-05
lr_scheduler_type: cosine
max_seq_length: 2048
max_steps: -1
num_train_epochs: 2
per_device_train_batch_size: 2
seed: 42
warmup_ratio: 0.1
```

## Citation
```
@misc{he2025guidingcomplexitymakesgood,
      title={Guiding Through Complexity: What Makes Good Supervision for Hard Math Reasoning Tasks?}, 
      author={Xuan He and Da Yin and Nanyun Peng},
      year={2025},
      eprint={2410.20533},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.20533}, 
}
```
