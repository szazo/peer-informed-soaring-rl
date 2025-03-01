from matplotlib.lines import Line2D
from typing import Optional, Union, cast, Any
from dataclasses import dataclass
import sigfig
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
@dataclass
class LinearRegressionResult:
    slope: float
    intercept: float
    r: float
    p: float
    slope_stderr: float
    intercept_stderr: float


class LinearRegressionPlot:

    _data: pd.DataFrame | np.ndarray | None
    _x: str | pd.Series | np.ndarray | None
    _y: str | pd.Series | np.ndarray | None

    def __init__(self,
                 data: pd.DataFrame | np.ndarray | None = None,
                 x: str | pd.Series | np.ndarray | None = None,
                 y: str | pd.Series | np.ndarray | None = None):

        self._data = data
        self._x = x
        self._y = y

    def _resolve_xy(
            self) -> tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]:
        assert (self._x is not None and self._y is not None
                and not isinstance(self._x, str)) or (
                    self._data is not None and isinstance(self._x, str)
                    and isinstance(self._y, str)), 'data or x,y is required'

        data = self._data
        x, y = self._x, self._y

        if data is not None:
            assert isinstance(x, str) and isinstance(
                y, str), 'x,y should be str key'
            return data[x], data[y]

        assert x is not None and y is not None, 'missing x or y'
        assert not isinstance(x, str) and not isinstance(
            y, str), 'x,y should not be str'

        return x, y

    def calculate_and_plot(self, **args):
        result = self.calculate()
        return self.plot(result, **args)

    def calculate(self) -> LinearRegressionResult:

        x, y = self._resolve_xy()

        result = stats.linregress(x=x, y=y)
        result = cast(Any, result)

        return LinearRegressionResult(slope=result.slope,
                                      intercept=result.intercept,
                                      r=result.rvalue,
                                      p=result.pvalue,
                                      slope_stderr=result.stderr,
                                      intercept_stderr=result.intercept_stderr)

    def plot(self,
             result: LinearRegressionResult,
             title_prefix: str,
             x_label: str,
             y_label: str,
             hue: Optional[Union[pd.Series, np.ndarray]] = None,
             style: Optional[Union[pd.Series, np.ndarray]] = None,
             out_filename_prefix: str | None = None,
             include_result_texts: bool = True,
             legend_title: str | None = None,
             show_n_in_legend: bool = True,
             x_lim: tuple[float, float] | None = None,
             y_lim: tuple[float, float] | None = None,
             x_err: float | np.ndarray | None = None,
             y_err: float | np.ndarray | None = None,
             equal_aspect: bool = True,
             figsize: tuple[float, float] = (11, 8),
             show_title: bool = True):

        # linear regression statistics
        fig, ax = plt.figure(figsize=figsize), plt.gca()
        if equal_aspect:
            ax.set_aspect('equal', adjustable='box')
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

        r_sigfig = sigfig.round(result.r,
                                sigfigs=3) if not np.isnan(result.r) else ''
        std_err_sigfig = sigfig.round(
            result.slope_stderr,
            sigfigs=3) if not np.isnan(result.slope_stderr) else 'nan'
        p_sigfig = sigfig.round(result.p,
                                sigfigs=2) if not np.isnan(result.p) else 'nan'

        x, _ = self._resolve_xy()

        n = np.shape(x)[0]
        detailed_title = title = f'{title_prefix} ($n={n}$);' + \
                f' \n$r={r_sigfig}$, ' + \
                f'slope stderr={std_err_sigfig}, p={p_sigfig} '

        if include_result_texts:
            title = detailed_title
        else:
            title = title_prefix

        sns.scatterplot(data=self._data,
                        x=self._x,
                        y=self._y,
                        ax=ax,
                        hue=hue,
                        style=style)
        # marker='',
        # facecolor='none')
        # alpha=0.5)

        x, y = self._resolve_xy()

        if x_err is not None or y_err is not None:
            ax.errorbar(
                x=x,
                y=y,
                xerr=x_err,
                yerr=y_err,
                fmt='none',
                ecolor='lightgray',
                zorder=0,
                elinewidth=0.5,
                # capsize=2,
                capthick=0.5)

        if show_title:
            ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        regression_line_y = x * result.slope + result.intercept

        sns.lineplot(x=x, y=regression_line_y, ax=ax, label='regression line')
        sns.lineplot(x=ax.get_xlim(),
                     y=ax.get_ylim(),
                     color='grey',
                     linestyle='--',
                     linewidth=1.0,
                     label='x=y')

        legend_handles, _ = ax.get_legend_handles_labels()
        if show_n_in_legend:
            legend_handles.append(Line2D([0], [0], color='w', label=f'n={n}'))

        ax.legend(title=legend_title, handles=legend_handles, loc='upper left')

        if out_filename_prefix is not None:
            fig.savefig(f'{out_filename_prefix}.png',
                        bbox_inches='tight',
                        pad_inches=0)
            fig.savefig(f'{out_filename_prefix}.svg',
                        bbox_inches='tight',
                        pad_inches=0)

        plt.show()

        return detailed_title
