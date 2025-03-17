from typing import Any
import numpy as np
from scipy import stats
from pathlib import Path
from functools import partial
from dataclasses import dataclass
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sigfig
from utils.plot_style import plot_style
from statannotations.Annotator import Annotator


@dataclass
class Info:
    name: str
    label: str | None = None
    hue: str | None = None
    more: dict[str, Any] | None = None
    filter_agents: list[str] | None = None


@dataclass(kw_only=True)
class Dataset(Info):
    filepath: Path


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    info: Info


class MultiAgentComparison:

    _single_teacher_150m_df: pd.DataFrame | None
    _single_teacher_350m_df: pd.DataFrame | None
    _student_with_teacher_350m_df: pd.DataFrame | None
    _student_without_teacher_350m_df: pd.DataFrame | None
    _student_with_teacher_150m_df: pd.DataFrame | None
    _student_without_teacher_150m_df: pd.DataFrame | None

    _experiment_name_column = 'eval_experiment'
    _hue_column = 'hue'

    def load2(self, datasets: list[Dataset]):

        loaded_datasets: list[LoadedDataset] = []
        for dataset in datasets:
            df = pd.read_csv(dataset.filepath)
            loaded = LoadedDataset(df=df,
                                   info=Info(
                                       name=dataset.name,
                                       hue=dataset.hue,
                                       label=dataset.label,
                                       more=dataset.more,
                                       filter_agents=dataset.filter_agents))
            loaded_datasets.append(loaded)

        return loaded_datasets

    def create_experiment_name_label_mapping(
            self, datasets: list[LoadedDataset]) -> dict[str, str]:

        result = {
            dataset.info.name:
            dataset.info.label
            if dataset.info.label is not None else dataset.info.name
            for dataset in datasets
        }
        return result

    def prepare_datasets_for_plotting(
            self, datasets: list[LoadedDataset]) -> pd.DataFrame:
        common_columns = [
            self._experiment_name_column, self._hue_column
        ] + self._scene_column_names + self._episode_column_names + [
            'training', 'velocity_earth_m_per_s_z', 'bird_name',
            'distance_from_core_m', 'reward', 'success', 'time_s',
            'time_s_without_lift', 'position_earth_m_z', 'distance',
            'combination'
        ]

        dfs = []
        for dataset in datasets:

            info = dataset.info
            # add experiment name column
            df = dataset.df.assign(**{self._experiment_name_column: info.name})
            df = df.assign(**{self._hue_column: info.hue})

            if info.more is not None:
                df = df.assign(**info.more)

            if info.filter_agents is not None and len(info.filter_agents) > 0:
                # filter agents
                df = df[df['agent_id'].isin(info.filter_agents)]

            # select common columns
            df = df[common_columns]
            assert isinstance(df, pd.DataFrame)
            dfs.append(df)

        # concat them
        df = pd.DataFrame(pd.concat(dfs))

        return df

    def _create_violinplot(self,
                           df: pd.DataFrame,
                           x: str,
                           x_label_mapping: dict[str, str],
                           y: str,
                           y_label: str,
                           figsize: tuple[float, float],
                           show: bool,
                           out_dir_path: Path,
                           palette: dict[str, Any] | str,
                           order: list[str],
                           success_rates: dict[str, float],
                           hue_field: str | None = None,
                           out_filename_prefix: str | None = None,
                           hline: float | None = None,
                           legend_title: str = '',
                           legend_loc: str = 'upper left',
                           linewidth=0.5):

        with plot_style():
            fig, ax = plt.figure(figsize=figsize), plt.gca()

            if hline is not None:
                ax.axhline(hline, color='gray', linestyle='--', linewidth=0.5)

            plt.text(x=2.35,
                     y=7.3,
                     s='150m\ninitial distance from thermal core',
                     horizontalalignment='right')

            plt.text(x=2.65,
                     y=7.3,
                     s='350m\ninitial distance from thermal core',
                     horizontalalignment='left')

            ax.axvline(2.5, color='gray', linestyle='--', linewidth=1)

            # hide x minor ticks
            ax.tick_params(axis='x', which='minor', bottom=False, top=False)
            ax.tick_params(axis='x', which='major', bottom=False, top=False)

            plotting_parameters: dict[str, Any] = dict(data=df,
                                                       x=x,
                                                       y=y,
                                                       density_norm='width',
                                                       hue='combination',
                                                       linewidth=linewidth,
                                                       gridsize=1000,
                                                       palette=palette,
                                                       order=order,
                                                       split=False)

            annotator_parameters = {**plotting_parameters}
            del annotator_parameters['hue']
            del annotator_parameters['palette']

            vp = sns.violinplot(ax=ax, **plotting_parameters)

            annotator_pairs = [
                ('teacher_alone_150m', 'student_alone_150m'),
                ('teacher_alone_150m', 'student_with_teachers_150m'),
                ('teacher_alone_350m', 'student_alone_350m'),
                ('student_alone_350m', 'student_with_teachers_350m'),
                ('teacher_alone_350m', 'student_with_teachers_350m'),
            ]

            legend = ax.legend(title=legend_title,
                               loc=legend_loc,
                               bbox_to_anchor=(0.78, 1.)).set_visible(False)

            annotator = Annotator(ax,
                                  pairs=annotator_pairs,
                                  **annotator_parameters)
            annotator.configure(test='Mann-Whitney', verbose=2,
                                line_width=0.5).apply_test().annotate(
                                    line_offset_to_group=0.12)

            ax.set_ylim(-2, 8.)

            ax.set_ylabel(y_label)

            current_ticks = vp.get_xticks()

            current_labels = [
                label.get_text() for label in vp.get_xticklabels()
            ]

            new_labels = [x_label_mapping[label] for label in current_labels]

            vp.set_xticks(list(range(len(new_labels))), new_labels)
            vp.set_xlabel('')

            y_pos = np.max(df[y]) + 1
            for i, experiment in enumerate(order):
                y_pos = vp.get_ylim()[0] + (vp.get_ylim()[1] -
                                            vp.get_ylim()[0]) * 0.015
                success_rate = success_rates[experiment]
                plt.text(x=i,
                         y=y_pos,
                         s=f'(success={success_rate})',
                         horizontalalignment='center')

            if out_filename_prefix is not None:
                fig.savefig(out_dir_path / f'{out_filename_prefix}.png',
                            bbox_inches='tight',
                            pad_inches=0,
                            dpi=300)
                fig.savefig(out_dir_path / f'{out_filename_prefix}.svg',
                            bbox_inches='tight',
                            pad_inches=0,
                            dpi=300)
                fig.savefig(out_dir_path / f'{out_filename_prefix}.eps',
                            bbox_inches='tight',
                            pad_inches=0)

            if show:
                plt.show()

    def create_stats_table(self, df: pd.DataFrame, latex_table_out_path: Path):

        experiment_groupby = df.groupby(self._experiment_name_column)

        per_experiment_stat_df = experiment_groupby.agg(
            v_z_mean=('velocity_earth_m_per_s_z', 'mean'),
            v_z_std=('velocity_earth_m_per_s_z', 'std'),
            distance=('distance', 'first'))

        assert isinstance(per_experiment_stat_df, pd.DataFrame)
        per_experiment_stat_df['v_z_sigfig'] = per_experiment_stat_df.apply(
            lambda row: self._sigfig_mean_std2(row['v_z_mean'], row['v_z_std']
                                               ),
            axis=1)

        per_episode_groupby = df.groupby([self._experiment_name_column] +
                                         self._scene_column_names +
                                         self._episode_column_names)

        per_episode_stat_df2 = per_episode_groupby.agg(
            reward=('reward', 'sum'),
            length=('reward', 'count'),
            success=('success', 'sum'),
            time_s=('time_s', 'max'),
            time_without_lift_s=('time_s_without_lift', 'max'),
        )

        episode_agg_stat_df2 = per_episode_stat_df2.groupby(
            self._experiment_name_column).agg({
                'success': ['sum'],
                'reward': ['mean', 'std'],
                'time_s': ['mean', 'std'],
                'time_without_lift_s': ['mean', 'std'],
            })

        assert isinstance(episode_agg_stat_df2, pd.DataFrame)

        episode_agg_stat_df2['reward_sigfig'] = episode_agg_stat_df2.apply(
            lambda row: self._sigfig_mean_std2(row[('reward', 'mean')], row[
                ('reward', 'std')]),
            axis=1)
        episode_agg_stat_df2['time_s_sigfig'] = episode_agg_stat_df2.apply(
            lambda row: self._sigfig_mean_std2(row[('time_s', 'mean')], row[
                ('time_s', 'std')]),
            axis=1)
        episode_agg_stat_df2[
            'time_without_lift_s_sigfig'] = episode_agg_stat_df2.apply(
                lambda row: self._sigfig_mean_std2(
                    row[('time_without_lift_s', 'mean')], row[
                        ('time_without_lift_s', 'std')]),
                axis=1)

        # flat the column names
        episode_agg_stat_df2.columns = [
            "_".join(a) for a in episode_agg_stat_df2.columns.to_flat_index()
        ]

        merged = pd.merge(per_experiment_stat_df,
                          episode_agg_stat_df2,
                          on='eval_experiment')

        result_df = merged[[
            'distance', 'success_sum', 'v_z_sigfig', 'reward_sigfig_'
        ]]
        result_df['success_sum'] = result_df['success_sum'] / 100.
        assert isinstance(result_df, pd.DataFrame)

        custom_sort = dict(teacher_alone_150m=0,
                           student_alone_150m=1,
                           student_with_teachers_150m=2,
                           teacher_alone_350m=3,
                           student_alone_350m=4,
                           student_with_teachers_350m=5)

        result_df = result_df.sort_values(by='eval_experiment',
                                          key=lambda x: x.map(custom_sort))

        latex_table = result_df.to_latex(
            index=True,
            escape=False,
            longtable=False,
            float_format='%.2f',
            columns=['success_sum', 'v_z_sigfig', 'reward_sigfig_'])

        with open(latex_table_out_path, "w") as f:
            f.write(latex_table)

        return result_df

    def _sigfig_mean_std(self, mean: float, std: float):
        return sigfig.round(mean, uncertainty=std, spacing=3, spacer=',')

    def _sigfig_mean_std2(self, mean: float, std: float):
        return sigfig.round(mean,
                            uncertainty=std,
                            spacing=3,
                            spacer=',',
                            cutoff=35)

    def _remove_outliers(self, field: str, group_df: pd.DataFrame):

        mode = 'z'

        if mode == 'iqr':

            q1 = group_df[field].quantile(0.25)
            q3 = group_df[field].quantile(0.75)

            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            return group_df[(group_df[field] >= lower_bound)
                            & (group_df[field] <= upper_bound)]
        elif mode == 'z':
            z_scores = stats.zscore(group_df[field])

            return group_df[(z_scores < 3) & (z_scores > -3)]

    def _downsample(self, group_df: pd.DataFrame):

        result_df = group_df[::100]

        return result_df

    def create_plots2(self,
                      df: pd.DataFrame,
                      experiment_name_label_mapping: dict[str, str],
                      out_dir_path: Path,
                      show: bool = True,
                      out_filename_prefix: str | None = None):

        success_rates = df.groupby('eval_experiment').agg(success=('success',
                                                                   'sum'))
        success_rates_dict: dict[str, float] = {
            index: (row['success'] / 100.)
            for index, row in success_rates.iterrows()
        }

        # remove outliers
        remove_func = partial(self._remove_outliers,
                              'velocity_earth_m_per_s_z')
        df = df.groupby(self._experiment_name_column,
                        as_index=False).apply(remove_func)

        df = df.groupby(self._experiment_name_column,
                        as_index=False).apply(self._downsample)

        pastel_palette = sns.color_palette("pastel")

        custom_palette = {
            "Single Agent": pastel_palette[0],
            "Multi Agent": pastel_palette[3]
        }
        custom_palette = {
            "teacher": pastel_palette[0],
            "student": pastel_palette[2],
            "student+teachers": pastel_palette[3],
            "student+students": pastel_palette[1]
        }

        order = [
            'teacher_alone_150m', 'student_alone_150m',
            'student_with_teachers_150m', 'teacher_alone_350m',
            'student_alone_350m', 'student_with_teachers_350m'
        ]

        mm = 1 / 25.4
        figsize = (200 * mm, 100 * mm)

        self._create_violinplot(df=df,
                                x=self._experiment_name_column,
                                x_label_mapping=experiment_name_label_mapping,
                                y='velocity_earth_m_per_s_z',
                                y_label='Vertical speed (m/s)',
                                hue_field=self._hue_column,
                                figsize=figsize,
                                palette=custom_palette,
                                show=show,
                                out_dir_path=out_dir_path,
                                out_filename_prefix='vertical_velocity',
                                order=order,
                                success_rates=success_rates_dict,
                                hline=0.)

    @property
    def _scene_column_names(self):
        scene_group_columns = ['thermal', 'rng_name', 'rng_state']
        return scene_group_columns

    @property
    def _episode_column_names(self):
        return ['episode']

    def _agent_column_names(self, df: pd.DataFrame):
        agent_name_column = None
        if 'agent_id' in df.columns:
            agent_name_column = 'agent_id'
        elif 'bird_name' in df.columns:
            agent_name_column = 'bird_name'

        assert agent_name_column is not None

        agent_column_names = ['agent_type', 'training']
        agent_column_names.append(agent_name_column)

        return agent_column_names


class MultiAgentStat:

    _obs_log: pd.DataFrame

    def load(self, file_path: str):

        self._obs_log = pd.read_csv(file_path)
        return self._obs_log

    def per_agent_vertical_velocity_stat(self):

        df = self._obs_log

        groupby = df.groupby(self._scene_column_names +
                             self._episode_column_names +
                             self._agent_column_names(df),
                             as_index=False)

        result_df = groupby.agg({
            'velocity_earth_m_per_s_z': ['mean', 'std', 'min', 'max'],
            'distance_from_core_m': ['mean', 'std', 'min', 'max'],
            'thermal_max_r_m': ['first'],
            'cutoff_reason': ['last']
        })

        result_df = pd.DataFrame(result_df[[
            'thermal', 'episode', 'agent_id', 'velocity_earth_m_per_s_z',
            'cutoff_reason', 'distance_from_core_m', 'thermal_max_r_m'
        ]])
        result_df = result_df.sort_values(
            by=['thermal', 'episode', 'agent_id'])
        result_df.to_csv('per_agent_vertical_velocity_stat.csv', index=False)

    @property
    def _scene_column_names(self):
        scene_group_columns = ['thermal', 'rng_name', 'rng_state']
        return scene_group_columns

    @property
    def _episode_column_names(self):
        return ['episode']

    def _agent_column_names(self, df: pd.DataFrame):
        agent_name_column = None
        if 'agent_id' in df.columns:
            agent_name_column = 'agent_id'
        elif 'bird_name' in df.columns:
            agent_name_column = 'bird_name'

        assert agent_name_column is not None

        agent_column_names = ['agent_type', 'training']
        agent_column_names.append(agent_name_column)

        return agent_column_names


multi_agent_stat = MultiAgentComparison()

multi_path = Path('results/eval/gaussian/multi_agent_policy')

out_dir_path = Path('results/stat/multi_agent')
out_dir_path.mkdir(parents=True, exist_ok=True)

student_filter = dict(filter_agents=['student0'])

datasets = [
    Dataset(name='teacher_alone_150m',
            more=dict(distance=150, combination='teacher'),
            hue='Single Agent',
            label='Teacher alone',
            filepath=multi_path / 'multi_agent_eval_teacher_alone_150m.csv'),
    Dataset(name='teacher_alone_350m',
            more=dict(distance=350, combination='teacher'),
            hue='Single Agent',
            label='Teacher alone',
            filepath=multi_path / 'multi_agent_eval_teacher_alone_350m.csv'),
    Dataset(name='student_alone_150m',
            more=dict(distance=150, combination='student'),
            hue='Single Agent',
            label='Student alone',
            filepath=multi_path / 'multi_agent_eval_student_alone_150m.csv',
            **student_filter),
    Dataset(name='student_alone_350m',
            more=dict(distance=350, combination='student'),
            hue='Single Agent',
            label='Student alone',
            filepath=multi_path / 'multi_agent_eval_student_alone_350m.csv',
            **student_filter),
    Dataset(name='student_with_teachers_150m',
            more=dict(distance=150, combination='student+teachers'),
            hue='Multi Agent',
            label='Student\nwith Teachers',
            filepath=multi_path /
            'multi_agent_eval_student_with_teachers_150m.csv',
            **student_filter),
    Dataset(name='student_with_teachers_350m',
            more=dict(distance=350, combination='student+teachers'),
            hue='Multi Agent',
            label='Student\nwith Teachers',
            filepath=multi_path /
            'multi_agent_eval_student_with_teachers_350m.csv',
            **student_filter),
]

loaded_datasets = multi_agent_stat.load2(datasets)
df = multi_agent_stat.prepare_datasets_for_plotting(loaded_datasets)
df.to_csv(out_dir_path / 'intermediate_multi_agent.csv')

experiment_name_label_mapping = multi_agent_stat.create_experiment_name_label_mapping(
    loaded_datasets)

multi_agent_stat.create_stats_table(df=df,
                                    latex_table_out_path=out_dir_path /
                                    "multi_agent_table.tex")

multi_agent_stat.create_plots2(
    df=df,
    experiment_name_label_mapping=experiment_name_label_mapping,
    show=True,
    out_dir_path=out_dir_path)
