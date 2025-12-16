import numpy as np
import matplotlib.pyplot as plt
# import scienceplots
from matplotlib.ticker import ScalarFormatter

class AcademicPlot:
    def __init__(self, style='seaborn-v0_8-paper', figsize=(6, 3.7)):
        """
        Initialize the plotting object.

        Parameters:
        style (str): Plot style.
        figsize (tuple): Figure size (width, height).
        """
        if style == 'ieee':
            try:
                plt.style.use(['science', 'ieee'])
                plt.rcParams['font.family'] = 'Times New Roman'
                self.style = style
            except (NameError, OSError):
                print("Please install `scienceplots` with `pip` to use the 'ieee' style.")
        else:
            self.style = style
        self.figsize = figsize
        plt.rcParams['font.family'] = 'Times New Roman'  # Set default font to Times New Roman

    
    def plot(self, x_list, y_list, labels = None, xlabel="X-axis", ylabel="Y-axis", 
            title="Title", filename=None, interp_x=None, linestyles=None, 
            colors=None, xlim=None, xticks=None, markers=None, scatter_data=None, 
            plot_grid = True, plot_legend =False, legend_loc = None, bbox_to_anchor = None):
        """
        Plot academic-style curves, supporting different x-axis resolutions, custom linestyles, colors, ranges, and labels.

        Parameters:
        x_list (list of array-like): X-axis data for multiple curves.
        y_list (list of array-like): Y-axis data for multiple curves.
        labels (list of str): Labels for each curve.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        title (str): Plot title.
        filename (str): Filename to save the figure. If None, do not save.
        interp_x (array-like): Unified x-axis for interpolation (optional).
        linestyles (list of str): Linestyle for each curve (optional).
        colors (list of str): Color for each curve (optional).
        xlim (tuple): X-axis range (xmin, xmax) (optional).
        xticks (list of float): Custom x-axis tick labels (optional).
        """
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)
        
        # Unify x-axis resolution (if needed)
        if interp_x is not None:
            new_y_list = []
            for x, y in zip(x_list, y_list):
                new_y_list.append(np.interp(interp_x, x, y))
            x_list = [interp_x] * len(new_y_list)
            y_list = new_y_list
        
        # Plot scatter data
        if scatter_data is not None:
            for scatter in scatter_data:
                ax.scatter(scatter['x'], scatter['y'],
                        label=scatter.get('label', None),
                        color=scatter.get('color', None),
                        marker=scatter.get('marker', 'o'),
                        s=scatter.get('size', 50),
                        alpha=0.6)  
                    
        # Plot multiple curves
        if labels is None:
            labels = [None]*len(x_list)
        for i, (x, y, label) in enumerate(zip(x_list, y_list, labels)):
            linestyle = linestyles[i] if linestyles and i < len(linestyles) else '-'
            color = colors[i] if colors and i < len(colors) else None
            marker = markers[i] if markers and i < len(markers) else None
            ax.plot(x, y, label=label, linestyle=linestyle, color=color, 
                    marker=marker, linewidth=1.0)
        
        # Set x-axis range
        if xlim:
            ax.set_xlim(xlim)
        
        # Set custom x-axis tick labels
        if xticks is not None:
            ax.set_xticks(xticks)
        
        # Set y-axis to scientific notation
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(formatter)

        # Optimize tick label position
        ax.tick_params(axis='y', pad=0)

        # Set axis labels and title
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=10)
        
        # Set grid and ticks
        if plot_grid is True:
            ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.7)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', length=4)
        
        # Add legend
        if plot_legend is True:
            if legend_loc is None:
                legend_loc = 'center'
            if bbox_to_anchor is None:
                bbox_to_anchor = (0.5, 1.1)
            # bbox_to_anchor: (x, y) - x controls horizontal position (0-1), y controls vertical position (>1 is above the plot)
            # ncol: number of columns in legend, columnspacing: space between columns
            # handlelength: line length in legend, handletextpad: space between legend text and symbol
            # borderpad: border padding, labelspacing: space between legend entries
            ax.legend(fontsize=6, loc=legend_loc, 
                     bbox_to_anchor=bbox_to_anchor,
                     frameon=True, edgecolor='black',
                     ncol=4, columnspacing=0.6,
                     handlelength=1.5, handletextpad=0.4,
                     borderpad=0.3, labelspacing=0.6)
            
        # Adjust margins
        # fig.tight_layout()
        
        # Save or show the figure
        if filename:
            plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.show()

    def plot_in_ax(self, x_list, y_list, ax, labels=None, xlabel="X-axis", ylabel="Y-axis", 
                  title=None, linestyles=None, colors=None, xlim=None, xticks=None, 
                  markers=None, scatter_data=None, plot_grid=True, plot_legend=False):
        """
        Plot curves in the specified axes.

        Parameters:
        ax : matplotlib.axes.Axes
            Axes object to plot on.
        Other parameters are the same as the plot method.
        """
        plt.style.use(self.style)
        
        # Plot scatter data
        if scatter_data is not None:
            for scatter in scatter_data:
                ax.scatter(scatter['x'], scatter['y'],
                        label=scatter.get('label', None),
                        color=scatter.get('color', None),
                        marker=scatter.get('marker', 'o'),
                        s=scatter.get('size', 50),
                        alpha=0.6)  
                    
        # Plot multiple curves
        if labels is None:
            labels = [None]*len(x_list)
        for i, (x, y, label) in enumerate(zip(x_list, y_list, labels)):
            linestyle = linestyles[i] if linestyles and i < len(linestyles) else '-'
            color = colors[i] if colors and i < len(colors) else None
            marker = markers[i] if markers and i < len(markers) else None
            ax.plot(x, y, label=label, linestyle=linestyle, color=color, 
                   marker=marker, linewidth=1.0)
        
        # Set x-axis range
        if xlim:
            ax.set_xlim(xlim)
        
        # Set custom x-axis tick labels
        if xticks is not None:
            ax.set_xticks(xticks)
        
        # Set y-axis to scientific notation
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(formatter)

        # Optimize tick label position
        ax.tick_params(axis='y', pad=0)

        # Set axis labels and title
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=10)
        
        # Set grid and ticks
        if plot_grid is True:
            ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.7)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', length=4)
        
        # Add legend
        if plot_legend is True:
            ax.legend(fontsize=6, loc='best', frameon=True, edgecolor='black')


    def eig_plot(self, mu, fig=None, ax=None,
                left=-6, right=0.5, ymin=-8, ymax=8, damping=0.05,
                line_width=0.5, s=40, dpi=300, figsize=None, base_color='black',
                show=True, latex=True, style='default', marker_style='x'
                ):
        """
        Plot utility for eigenvalues in the S domain.

        Parameters
        ----------
        mu : array, optional
            an array of complex eigenvalues
        fig : figure handle, optional
            existing matplotlib figure handle
        ax : axis handle, optional
            existing axis handle
        left : int, optional
            left tick for the x-axis, by default -6
        right : float, optional
            right tick, by default 0.5
        ymin : int, optional
            bottom tick, by default -8
        ymax : int, optional
            top tick, by default 8
        damping : float, optional
            damping value for which the dash plots are drawn
        line_width : float, optional
            default line width, by default 0.5
        s : float or array-like, shape (n, ), optional
            The marker size in points**2
        dpi : int, optional
            figure dpi
        figsize : [type], optional
            default figure size, by default None
        base_color : str, optional
            base color for negative eigenvalues
        show : bool, optional
            True to show figure after plot, by default True
        latex : bool, optional
            True to use latex, by default True

        Returns
        -------
        figure
            matplotlib figure object
        axis
            matplotlib axis object

        """
        config_tol = 1e-4
        plt.style.use(self.style)

        mu_real = mu.real
        mu_imag = mu.imag
        p_mu_real, p_mu_imag = list(), list()
        z_mu_real, z_mu_imag = list(), list()
        n_mu_real, n_mu_imag = list(), list()

        for re, im in zip(mu_real, mu_imag):
            if abs(re) <= config_tol:
                z_mu_real.append(re)
                z_mu_imag.append(im)
            elif re > config_tol:
                p_mu_real.append(re)
                p_mu_imag.append(im)
            elif re < -config_tol:
                n_mu_real.append(re)
                n_mu_imag.append(im)

        if figsize is None:
            figsize = self.figsize
        if fig is None or ax is None:
            fig = plt.figure(dpi=dpi, figsize=figsize)
            ax = plt.gca()
        # Change marker style for scatter plot
        if marker_style == 'o':
            # Use hollow circle
            ax.scatter(z_mu_real, z_mu_imag, marker='o', s=s, linewidth=0.5, 
                    facecolors='none', edgecolors='green')
            ax.scatter(n_mu_real, n_mu_imag, marker='o', s=s, linewidth=0.5, 
                    facecolors='none', edgecolors=base_color)
            ax.scatter(p_mu_real, p_mu_imag, marker='o', s=s, linewidth=0.5, 
                    facecolors='none', edgecolors='red')
        else:
            # Use X marker (default)
            ax.scatter(z_mu_real, z_mu_imag, marker='x', s=s, linewidth=1.0, 
                    facecolors='none', edgecolors='green')
            ax.scatter(n_mu_real, n_mu_imag, marker='x', s=s, linewidth=1.2, 
                    color=base_color)
            ax.scatter(p_mu_real, p_mu_imag, marker='x', s=s, linewidth=1.0, 
                    color='red')

        # axes lines
        ax.axhline(linewidth=0.5, color='grey', linestyle='--')
        ax.axvline(linewidth=0.5, color='grey', linestyle='--')

        # TODO: Improve the damping and range
        # --- plot 5% damping lines ---
        xin = np.arange(left, 0, 0.01)
        yneg = xin / damping
        ypos = - xin / damping

        ax.plot(xin, yneg, color='grey', linewidth=line_width, linestyle='--')
        ax.plot(xin, ypos, color='grey', linewidth=line_width, linestyle='--')
        # --- damping lines end ---

        if latex:
            ax.set_xlabel('Real [$s^{-1}$]')
            ax.set_ylabel('Imaginary [$s^{-1}$]')
        else:
            ax.set_xlabel('Real [s -1]')
            ax.set_ylabel('Imaginary [s -1]')

        ax.set_xlim(left=left, right=right)
        ax.set_ylim(ymin, ymax)

        if show is True:
            plt.show()
        return fig, ax


def _enforce_ratio(goal_ratio, supx, infx, supy, infy):
    """
    Computes the right value of `supx,infx,supy,infy` to obtain the desired
    ratio in :func:`plot_eigs`. Ratio is defined as
    ::
        dx = supx - infx
        dy = supy - infy
        max(dx,dy) / min(dx,dy)

    :param float goal_ratio: the desired ratio.
    :param float supx: the old value of `supx`, to be adjusted.
    :param float infx: the old value of `infx`, to be adjusted.
    :param float supy: the old value of `supy`, to be adjusted.
    :param float infy: the old value of `infy`, to be adjusted.
    :return tuple: a tuple which contains the updated values of
        `supx,infx,supy,infy` in this order.
    """

    dx = supx - infx
    if dx == 0:
        dx = 1.0e-16
    dy = supy - infy
    if dy == 0:
        dy = 1.0e-16
    ratio = max(dx, dy) / min(dx, dy)

    if ratio >= goal_ratio:
        if dx < dy:
            goal_size = dy / goal_ratio

            supx += (goal_size - dx) / 2
            infx -= (goal_size - dx) / 2
        elif dy < dx:
            goal_size = dx / goal_ratio

            supy += (goal_size - dy) / 2
            infy -= (goal_size - dy) / 2

    return (supx, infx, supy, infy)

def plot_eigs(
    dmd_eigs,
    show_axes=True,
    show_unit_circle=True,
    figsize=(8, 8),
    title="",
    narrow_view=False,
    dpi=None,
    filename=None,
    show = True
):
    """
    Plot the eigenvalues.

    :param dmd: DMD instance.
    :type dmd: pydmd.DMDBase
    :param bool show_axes: if True, the axes will be showed in the plot.
        Default is True.
    :param bool show_unit_circle: if True, the circle with unitary radius
        and center in the origin will be showed. Default is True.
    :param tuple(int,int) figsize: tuple in inches defining the figure
        size. Default is (8, 8).
    :param str title: title of the plot.
    :param narrow_view bool: if True, the plot will show only the smallest
        rectangular area which contains all the eigenvalues, with a padding
        of 0.05. Not compatible with `show_axes=True`. Default is False.
    :param dpi int: If not None, the given value is passed to
        ``plt.figure``.
    :param str filename: if specified, the plot is saved at `filename`.
    """

    if dmd_eigs is None:
        raise ValueError(
            "The eigenvalues have not been computed."
            "You have to call the fit() method."
        )

    if dpi is not None:
        plt.figure(figsize=figsize, dpi=dpi)
    else:
        plt.figure(figsize=figsize)

    plt.title(title)
    plt.gcf()
    ax = plt.gca()

    points = ax.plot(dmd_eigs.real, dmd_eigs.imag, "bo", label="Eigenvalues")

    if narrow_view:
        supx = max(dmd_eigs.real) + 0.05
        infx = min(dmd_eigs.real) - 0.05

        supy = max(dmd_eigs.imag) + 0.05
        infy = min(dmd_eigs.imag) - 0.05

        supx, infx, supy, infy = _enforce_ratio(8, supx, infx, supy, infy)

        # set limits for axis
        ax.set_xlim((infx, supx))
        ax.set_ylim((infy, supy))

        # x and y axes
        if show_axes:
            endx = np.min([supx, 1.0])
            ax.annotate(
                "",
                xy=(endx, 0.0),
                xytext=(np.max([infx, -1.0]), 0.0),
                arrowprops=dict(arrowstyle=("->" if endx == 1.0 else "-")),
            )

            endy = np.min([supy, 1.0])
            ax.annotate(
                "",
                xy=(0.0, endy),
                xytext=(0.0, np.max([infy, -1.0])),
                arrowprops=dict(arrowstyle=("->" if endy == 1.0 else "-")),
            )
    else:
        # set limits for axis
        limit = np.max(np.ceil(np.absolute(dmd_eigs)))
        ax.set_xlim((-limit, limit))
        ax.set_ylim((-limit, limit))

        # x and y axes
        if show_axes:
            ax.annotate(
                "",
                xy=(np.max([limit * 0.8, 1.0]), 0.0),
                xytext=(np.min([-limit * 0.8, -1.0]), 0.0),
                arrowprops=dict(arrowstyle="->"),
            )
            ax.annotate(
                "",
                xy=(0.0, np.max([limit * 0.8, 1.0])),
                xytext=(0.0, np.min([-limit * 0.8, -1.0])),
                arrowprops=dict(arrowstyle="->"),
            )

    plt.ylabel("Imaginary part")
    plt.xlabel("Real part")

    if show_unit_circle:
        unit_circle = plt.Circle(
            (0.0, 0.0),
            1.0,
            color="green",
            fill=False,
            label="Unit circle",
            linestyle="--",
        )
        ax.add_artist(unit_circle)

    # Dashed grid
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle("-.")
    ax.grid(True)

    # legend
    if show_unit_circle:
        ax.add_artist(
            plt.legend(
                [points, unit_circle],
                ["Eigenvalues", "Unit circle"],
                loc="best",
            )
        )
    else:
        ax.add_artist(plt.legend([points], ["Eigenvalues"], loc="best"))

    ax.set_aspect("equal")

    if filename:
        plt.savefig(filename)
    if show is True:
        plt.show()

import seaborn as sns

def plot_participation_factors(participation_factors):
    """
    Plot a heatmap of participation factors (without value annotations).
    
    Parameters:
    participation_factors: Participation factor matrix
    """
    # Take the absolute value of participation factors
    pf_abs = np.abs(participation_factors)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pf_abs, 
                cmap='YlOrRd',
                xticklabels=[f'Mode {i+1}' for i in range(pf_abs.shape[1])],
                yticklabels=[f'State {i+1}' for i in range(pf_abs.shape[0])],
                cbar_kws={'label': 'Participation Factor'})
    plt.title('Participation Factors')
    plt.xlabel('Modes')
    plt.ylabel('States')
    plt.tight_layout()
    plt.show()
