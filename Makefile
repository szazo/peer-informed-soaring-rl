unittest:
	python -m pytest --disable-warnings -s

plot_gaussian_thermal:
	python run_plot_thermal.py show_plots_in_cli=true

train_single:
	python run_train.py --config-dir config --config-name 'train_config'

train_multirun:
	python run_train.py --config-dir config --multirun --config-name 'train_multirun_config'

generate_gaussian_thermal_eval_configs:
	python tools/eval/generate_eval_configs.py --seed=43242378742983795 --count=100

train_two_level_multi_agent:
	python run_train.py --config-dir config --config-name 'two_level_multi_agent_train_config' model.non_learning.model_source_logger.with_existing_id=$(POLICY_NEPTUNE_ID) train_env.env.params.max_sequence_length=$(SEQ_LENGTH) train_env.env.params.max_closest_agent_count=$(AGENT_SEQ_LENGTH) 

eval_multi_agent_student_all: eval_multi_agent_student_with_teachers_350m eval_multi_agent_student_alone_350m eval_multi_agent_student_with_teachers_150m eval_multi_agent_student_alone_150m

eval_multi_agent_teacher_all: eval_multi_agent_teacher_alone_150m eval_multi_agent_teacher_alone_350m

eval_multi_agent_student_with_teachers_350m:
	python run_eval.py --config-dir config --config-name 'multi_agent_eval_student_with_teachers_350m' --multirun hydra.callbacks.merge_observation_logs.output.filename_prefix="multi_agent_eval_student_with_teachers_350m" model_source_logger.with_existing_id=$(POLICY_NEPTUNE_ID) checkpoint_name=$(POLICY_CHECKPOINT_NAME) evaluators.main.env.env.params.max_sequence_length=$(SEQ_LENGTH) evaluators.main.env.env.params.max_closest_agent_count=$(AGENT_SEQ_LENGTH)

eval_multi_agent_student_alone_350m:
	python run_eval.py --config-dir config --config-name 'multi_agent_eval_student_alone_350m' --multirun hydra.callbacks.merge_observation_logs.output.filename_prefix="multi_agent_eval_student_alone_350m" model_source_logger.with_existing_id=$(POLICY_NEPTUNE_ID) checkpoint_name=$(POLICY_CHECKPOINT_NAME) evaluators.main.env.env.params.max_sequence_length=$(SEQ_LENGTH) evaluators.main.env.env.params.max_closest_agent_count=$(AGENT_SEQ_LENGTH)

eval_multi_agent_student_with_teachers_150m:
	python run_eval.py --config-dir config --config-name 'multi_agent_eval_student_with_teachers_150m' --multirun hydra.callbacks.merge_observation_logs.output.filename_prefix="multi_agent_eval_student_with_teachers_150m" model_source_logger.with_existing_id=$(POLICY_NEPTUNE_ID) checkpoint_name=$(POLICY_CHECKPOINT_NAME) evaluators.main.env.env.params.max_sequence_length=$(SEQ_LENGTH) evaluators.main.env.env.params.max_closest_agent_count=$(AGENT_SEQ_LENGTH)

eval_multi_agent_student_alone_150m:
	python run_eval.py --config-dir config --config-name 'multi_agent_eval_student_alone_150m' --multirun hydra.callbacks.merge_observation_logs.output.filename_prefix="multi_agent_eval_student_alone_150m" model_source_logger.with_existing_id=$(POLICY_NEPTUNE_ID) checkpoint_name=$(POLICY_CHECKPOINT_NAME) evaluators.main.env.env.params.max_sequence_length=$(SEQ_LENGTH) evaluators.main.env.env.params.max_closest_agent_count=$(AGENT_SEQ_LENGTH)

eval_multi_agent_teacher_alone_150m:
	python run_eval.py --config-dir config --config-name 'multi_agent_eval_teacher_alone_150m' --multirun  hydra.callbacks.merge_observation_logs.output.filename_prefix="multi_agent_eval_teacher_alone_150m" model_source_logger.with_existing_id=$(POLICY_NEPTUNE_ID) evaluators.main.env.env.params.max_sequence_length=$(SEQ_LENGTH)

eval_multi_agent_teacher_alone_350m:
	python run_eval.py --config-dir config --config-name 'multi_agent_eval_teacher_alone_350m' --multirun  hydra.callbacks.merge_observation_logs.output.filename_prefix="multi_agent_eval_teacher_alone_350m" model_source_logger.with_existing_id=$(POLICY_NEPTUNE_ID) evaluators.main.env.env.params.max_sequence_length=$(SEQ_LENGTH)

play_observation_log:
	python run_custom_job.py --config-dir config --config-name 'play_observation_log_config' job.observation_log_filepath=${OBS_LOG_FILEPATH}
