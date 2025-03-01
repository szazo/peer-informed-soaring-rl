import scienceplots
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator


def violinplot(df: pd.DataFrame,
               x_field: str,
               y_field: str,
               hue_field: str,
               figsize: tuple[int, int],
               title: str,
               x_label: str,
               y_label: str,
               hline: float | None,
               legend_title: str,
               legend_replace: dict[str, str],
               annotator_pairs: list[tuple],
               order: list[str] | None = None,
               hue_order: list[str] | None = None,
               legend_loc: str = 'upper left',
               out_filename_prefix: str | None = None,
               bw: float = 0.2,
               cut: float = 0.5,
               show_title: bool = True):

    # with plt.style.context(['science']):
    fig, ax = plt.figure(figsize=figsize), plt.gca()
    if hline is not None:
        ax.axhline(hline, color='gray', linestyle='--', linewidth=1)

    # old seaborn does not found fields in the index
    df = df.reset_index()

    plotting_parameters = {
        'data': df,
        'x': x_field,
        'y': y_field,
        'order': order,
        'hue': hue_field,
        'hue_order': hue_order,
        'cut': cut,
        'bw': bw,
        # 'bw': 'silverman',
        'gridsize': 1000,
        'linewidth': .5
    }

    vp = sns.violinplot(ax=ax, **plotting_parameters)

    annotator = Annotator(ax, pairs=annotator_pairs, **plotting_parameters)
    annotator.configure(test='Mann-Whitney', verbose=2, line_width=0.5)
    annotator.apply_test()
    annotator.annotate(line_offset_to_group=0.12)

    legend = ax.legend(title=legend_title, loc=legend_loc)

    # replace legend texts
    for text in legend.get_texts():
        found = legend_replace[text.get_text()]
        text.set_text(found)

    # set title, labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if show_title:
        ax.set_title(title)

    # hide x minor ticks
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.tick_params(axis='x', which='major', bottom=True, top=False)

    counts = df.groupby([x_field, hue_field]).size()
    counts = counts.unstack().reset_index()

    y_pos = np.max(df[y_field]) + 1
    for _, row in counts.iterrows():
        x_value = row[x_field]
        thermal_index = np.where(counts[x_field] == x_value)[0][0]
        # y_pos = np.max(df.groupby(x_field).get_group(x_value)[y_field])
        y_pos = vp.get_ylim()[0] + (vp.get_ylim()[1] -
                                    vp.get_ylim()[0]) * 0.015
        n = row['ai']
        assert n == row['bird'], 'ai n should be equal to bird n'
        plt.text(x=thermal_index,
                 y=y_pos,
                 s=f'(n={n})',
                 horizontalalignment='center')

    if out_filename_prefix is not None:
        fig.savefig(f'{out_filename_prefix}.png',
                    bbox_inches='tight',
                    pad_inches=0)
        fig.savefig(f'{out_filename_prefix}.svg',
                    bbox_inches='tight',
                    pad_inches=0)

    plt.show()
