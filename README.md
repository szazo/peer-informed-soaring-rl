# Collective Thermal Soaring of Gliders Based on Peer-Information and Transformer-Driven Reinforcement Learning

## Creating and activating virtual env

1. Create a *virtual env* which uses Python 3.11 (**replace** `VENV_DIR` e.g. to `~/.envs/soaringrl`):
    ```bash
    pyenv install 3.11
    pyenv shell 3.11
    python -m venv VENV_DIR
    ```
2. Activate the virtual env:
    ```bash
    source VENV_DIR/bin/activate
    ```
    
## Initialize the project for development

1. Enter into **project root**
2. Start developing with **editable** (or **develop**) mode:
    ```bash
    python -m pip install --editable ".[dev]"
    ```
3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Experiment logger

For logging the experiments and storing trained weights https://neptune.ai/[neptune.ai] is used.

Create a new project and:
1. set the project id in `config/logger/neptune_logger.yaml` after `project:`
2. set the `NEPTUNE_API_TOKEN` environment variable e.g. in `.bashrc`, `.zshrc` (it can also be set in the config, but not recommended):

```bash
export NEPTUNE_API_TOKEN=TOKEN-VALUE
```

# Plot gaussian thermals

Just for checking for generating a Gaussian thermal plot

```
make plot_gaussian_thermal
```

# Single-Agent training

## Single training

Train using default parameters:

```bash
make train_single
```

## Run train with multiple parameters

Run multiple training jobs with different sequence lengths:

```bash
make train_multirun
```

# Multi-Agent training

- Set `POLICY_NEPTUNE_ID` to the Neptune Run ID which contains the trained single agent teacher policy.
- Set `SEQ_LENGTH` to the sequence length matches the trained teacher policy.
- Set `AGENT_SEQ_LENGTH` to the number of agents student use trajectories from (including itself).

```bash
POLICY_NEPTUNE_ID=<NEPTUNE RUN ID> SEQ_LENGTH=<time seq. length> AGENT_SEQ_LENGTH=<agent seq. length> make train_two_level_multi_agent
```

For example:

```bash
POLICY_NEPTUNE_ID=GLID-4517 SEQ_LENGTH=50 AGENT_SEQ_LENGTH=5 make train_two_level_multi_agent
```


# Evaluate

## Generate reproducible eval configs

First, generate reproducible evaluation configurations for gaussian thermals:

(You can change the initial seed and the count in the `Makefile`)

```bash
make generate_gaussian_thermal_eval_configs
```

## Teacher Evaluation

Evaluate the trained teacher policy using the reproducible configs.

- Set `POLICY_NEPTUNE_ID` to the Neptune Run ID which contains the trained policy.
- Set `SEQ_LENGTH` to the sequence length matches the policy.

```bash
POLICY_NEPTUNE_ID=<NEPTUNE RUN ID> SEQ_LENGTH=<time seq. length> make eval_multi_agent_teacher_all
```

For example:
```
POLICY_NEPTUNE_ID=GLID-4517 SEQ_LENGTH=50 make eval_multi_agent_teacher_all
```

Observation logs will be created under `results/eval/gaussian/multi_agent_policy`.


## Student Evaluation

Evaluate the trained teacher policy using the reproducible configs.

- Set `POLICY_NEPTUNE_ID` to the Neptune Run ID which contains the trained policy.
- Set `POLICY_CHECKPOINT_NAME` to the checkpoint of the run (under `checkpoint` key) to be used e.g. `weights42` (select e.g. based on `checkpoint/mean_minus_std` metrics)
- Set `SEQ_LENGTH` to the sequence length matches the policy.
- Set `AGENT_SEQ_LENGTH` to the agent sequence length matches the policy.

```bash
POLICY_NEPTUNE_ID=<NEPTUNE RUN ID> POLICY_CHECKPOINT_NAME=<checkpoint> SEQ_LENGTH=<time seq. length> AGENT_SEQ_LENGTH=<agent seg. length> make eval_multi_agent_student_all
```

For example:
```
POLICY_NEPTUNE_ID=GLID-4924 POLICY_CHECKPOINT_NAME=weights108 SEQ_LENGTH=50 AGENT_SEQ_LENGTH=5 make eval_multi_agent_student_all
```

Observation logs will be created under `results/eval/gaussian/multi_agent_policy`.

# Play observation logs

The observations can be played in the 3D interactive player using e.g. the following command:

```
 OBS_LOG_FILEPATH=results/eval/gaussian/multi_agent_policy/multi_agent_eval_student_with_teachers_350m.csv  make play_observation_log
```

More settings can be set in `config/play_observation_log_config.yaml`
