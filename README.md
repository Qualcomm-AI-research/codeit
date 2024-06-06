# CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay

This repository contains the official implementation of
[CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay](https://arxiv.org/abs/2402.04858)
by Natasha Butt, Blazej Manczak, Auke Wiggers, Corrado Rainone, [David Zhang](https://davzha.netlify.app), [Michaël Defferrard](https://deff.ch), and [Taco Cohen](https://tacocohen.wordpress.com),
published at ICML 2024.

Contact: <n.e.butt@uva.nl>.
Work completed while at QUVA Lab as part of the research collaboration between Qualcomm Technologies, Inc. (QTI) and the University of Amsterdam (UvA).

## Abstract

> Large language models are increasingly solving tasks that are commonly believed to require human-level reasoning ability. However, these models still perform very poorly on benchmarks of general intelligence such as the Abstraction and Reasoning Corpus (ARC). In this paper, we approach ARC as a programming-by-examples problem, and introduce a novel and scalable method for language model self-improvement called Code Iteration (CodeIt). Our method iterates between 1) program sampling and hindsight relabeling, and 2) learning from prioritized experience replay. By relabeling the goal of an episode (i.e., the target program output given input) to the realized output produced by the sampled program, our method effectively deals with the extreme sparsity of rewards in program synthesis. Applying CodeIt to the ARC dataset, we demonstrate that prioritized hindsight replay, along with pre-training and data-augmentation, leads to successful inter-task generalization. CodeIt is the first neuro-symbolic approach that scales to the full ARC evaluation dataset. Our method solves 15% of ARC evaluation tasks, achieving state-of-the-art performance and outperforming existing neural and symbolic baselines. Our code is available at https://github.com/Qualcomm-AI-research/codeit.

## Getting started

Clone the repository.

```bash
git clone https://github.com/Qualcomm-AI-research/codeit.git
cd codeit
```

Build the Docker image from inside the checked-out repository. On Linux, this can be done using

```bash
docker build . -t codeit:latest
```

Once the image has built successfully, we can run a container that is based on it.
On Linux, the command that does this and also

- mounts the current working directory into the container
- changes to the current working directory
- exposes all of the host's GPUs in the container

is

```bash
docker run --rm -it -v "${PWD}":"${PWD}" -w "${PWD}" --gpus=all codeit:latest /bin/bash
```

Note: we require Python 3.9 at least due to our use of the ast library.

### Raw data processing, optional

Note: the stored [`data/training`](data/training) and [`data/evaluation`](data/evaluation) directories were created by the below.

Download official ARC [training](https://github.com/fchollet/ARC/tree/master/data/training) and [evaluation](https://github.com/fchollet/ARC/tree/master/data/evaluation) data into `data/raw`:

```bash
bash download-data.sh
```

Preprocess ARC data:

```bash
python run/preprocess_tasks.py
```

## Running experiments

### Running the mutation baseline

Mutate ARC training tasks and save these as `outputs/mutation/mutated_tasks_train_{i}.json`:

```bash
mkdir -p outputs/mutation/
python run/run_mutate.py hydra.run.dir=outputs/mutation/
```

Note: there will be many errors printed to the log as the mutation algorithm tests code by executing with timeout and printing errors. The mutation algorithm stops after `baseline.num_samples` samples.

To evaluate mutated programs from the mutated tasks by executing each program on all ARC evaluation tasks, run:

```bash
python run/evaluate_mutations.py hydra.run.dir=outputs/mutation/ baseline.tasks_file=outputs/mutation/mutated_tasks_train_ baseline.parallel=True baseline.sig_alarm=True
```

### Running CodeIt

```bash
mkdir -p outputs/codeit/
python run/run_codeit.py hydra.run.dir=outputs/codeit/
```

Note: by default, CodeIt will use the mutated tasks stored in `data/mutated_tasks_train_*.json`.
To instead use the mutated tasks created by [the above](#running-the-mutation-baseline), set `replay_buffer.mutated_train_tasks_file=outputs/mutation/mutated_tasks_train_`.

Note: to reduce memory requirements, lower `data.dataloader.batch_size` or consider using a smaller model such as `codet5_small`.

Outputs saved to `outputs/codeit/`:
- `performance.csv` contains performance and number of programs/tasks in buffer for every meta-iteration
- `tensorboard` directories contain scalar metrics and text logs for all programs with seen example performance of 1, along with their task demonstration performance and test performance
- `log_{i}.json`: policy samples at meta-iteration `i`
- `solutions_{i}.json`: solutions at meta-iteration `i`
- `last.ckpt.dir`: last model checkpoint
- `log.out`: terminal output

### Running ablations

A1, no ExIt: \
first run the [mutation baseline](#running-the-mutation-baseline) to obtain mutated programs evaluated on ARC evaluation tasks, then
```bash
mkdir -p outputs/a1_no_exit/
python run/run_codeit.py hydra.run.dir=outputs/a1_no_exit/ ablation.used=True exit.add_policy_samples=False
```

A2, no relabeling:
```bash
mkdir -p outputs/a2_no_relabeling/
python run/run_codeit.py hydra.run.dir=outputs/a2_no_relabeling/ ablation.sampling_and_filtering=True
```

A3, no priority:
```bash
mkdir -p outputs/a3_no_priority/
python run/run_codeit.py hydra.run.dir=outputs/a3_no_priority/ replay_buffer.performance_penalty=0
```

A4, no pretraining:
```bash
mkdir -p outputs/a4_no_pretraining/
python run/run_codeit.py hydra.run.dir=outputs/a4_no_pretraining/ model.random_weights=True
```

A5, one demo:
```bash
mkdir -p outputs/a5_one_demo_example/
python run/run_codeit.py hydra.run.dir=outputs/a5_one_demo_example/ data.dataloader.tokenizer.n_examples=1
```

A6, no mutation:
```bash
mkdir -p outputs/a6_no_mutation/
python run/run_codeit.py hydra.run.dir=outputs/a6_no_mutation/ replay_buffer.mutated_train_tasks_file=""
```

### Analyzing results

Use [`results.ipynb`](results.ipynb) to compute the mutation baseline performance, CodeIt cumulative performance, and CodeIt policy-only performance.

Note: if you ran CodeIt with the mutated tasks stored in `data/`, the default, leave `MUTATION_PATH=data/`.
Else, if you re-ran the [mutation baseline](#running-the-mutation-baseline) and ran CodeIt with mutated tasks from this run (by setting `replay_buffer.mutated_train_tasks_file=outputs/mutation/mutated_tasks_train_`), set `MUTATION_PATH=outputs/mutation/`.

## Repository structure

```text
├─ codeit/
│   ├─ augment/
│   │   ├─ genetic.py                           # Mutation algorithm implementation
│   │   ├─ mutate_grid.py                       # Lower level functions for input mutation (not used for mutation baseline)
│   │   ├─ mutate_program.py                    # Lower level functions for program mutation
│   │   ├─ program_sampler.py                   # Random program sampler implementation (reuses code from mutation algorithm)
│   │   └─ type_inference.py                    # Code for type inference
│   ├─ dsl/                                     # Domain-specific language, based on https://github.com/michaelhodel/arc-dsl
│   │   ├─ arc_types.py                         # Type system
│   │   ├─ dsl.py                               # Primitive function definitions
│   │   ├─ primitives.py                        # Primitive constant definitions and function names
│   │   └─ solvers.py                           # Solutions to ARC training tasks
│   ├─ policy/
│   │   ├─ environment.py                       # Code for executing programs with timeout
│   │   ├─ inference.py                         # Code for policy generation and evaluation
│   │   └─ tokenize.py                          # Code for tokenizing examples
│   ├─ agent.py                                 # Agent manages the replay buffer, samples the policy, adds to solutions
│   ├─ exit_data_module.py                      # Exit data module samples the replay buffer and collates dataset for dataloaders
│   ├─ replay_buffer.py                         # Stores tokenized training examples
│   ├─ task.py                                  # Functionality related to tasks
│   ├─ callbacks.py                             # Contains callback for model saving
│   ├─ utils.py                                 # General purpose functions
│   ├─ typing_custom.py                         # Contains custom typing functions
│   ├─ concept_arc.py                           # Functions for loading and processing concept arc dataset
│   └─ hf_model_module.py                       # Pytorch lightning module
├─ data/
│   ├─ evaluation/                              # Processed ARC evaluation tasks
│   ├─ training/                                # Processed ARC training tasks
│   ├─ codeit_arc_eval_solutions_100.json       # Solutions after 100 meta-iterations for codeit run
│   ├─ mutation_performance.csv                 # Mutation baseline d1 performance
│   ├─ mutation_solutions_19200.json            # Mutation baseline d1 solutions after 19,200 samples
│   ├─ mutated_tasks_train_9600.json            # Mutation baseline d1 tasks after 9,600 samples
│   ├─ mutated_tasks_train_19200.json           # Mutation baseline d1 tasks after 19,200 samples
│   └─ split_keys.json                          # Contains custom validation split for training tasks
├─ outputs/                                     # Experiment outputs, created by running the above commands
├─ run/
│   ├─ config/
│   │   ├─ models/                              # Configs for various models
│   │   └─ base_config.yaml                     # All default parameters are stored here
│   ├─ preprocess_tasks.py                      # Script for preprocessing ARC tasks
│   ├─ run_codeit.py                            # Script for running codeit
│   ├─ evaluate_mutations.py                    # Script for evaluating mutated tasks
│   └─ run_mutate.py                            # Script for mutating programs
├─ Dockerfile
├─ download-data.sh                             # Script to download raw ARC data
├─ LICENSE
├─ README.md
├─ requirements.txt
├─ results.ipynb                                # Notebook for reporting performance of CodeIt and mutation baseline
└─ setup.py
```

The domain-specific language for solving ARC, implemented in [`codeit/dsl`](codeit/dsl), is largely based on [Michael Hodel's implementation](https://github.com/michaelhodel/arc-dsl), with some minor updates for ease of program execution and mutation.

## Citation

If you find our code useful, please cite:

```text
@inproceedings{butt2024codeit,
  title = {CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay},
  author = {Butt, Natasha and Manczak, Blazej and Wiggers, Auke and Rainone, Corrado and Zhang, David and Defferrard, Micha{\"e}l and Cohen, Taco},
  booktitle = {International Conference on Machine Learning},
  year = {2024},
  organization = {PMLR},
  eprint = {2402.04858},
  url = {https://arxiv.org/abs/2402.04858},
}
```
