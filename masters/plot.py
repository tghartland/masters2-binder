import math
import numpy as np
import logging
logging.basicConfig(level=logging.WARN)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator

y_limits = {
    "q*": (1e-4, 4),
    "qbh": (1e-5, 1e-1),
    "wprime": (5e-5, 4),
}

particle_symbol = {
    "q*": "q*",
    "qbh": "QBH",
    "wprime": "W'",
}

rapidity = {
    "q*": 0.6,
    "qbh": 0.6,
    "wprime": 0.6,
}

class PlusMinus(object):
    pass


class PlusMinusHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch1 = mpatches.Rectangle([x0, y0], width, height, facecolor='#ffff00',
                                   transform=handlebox.get_transform())
        patch2 = mpatches.Rectangle([x0, y0+height/4], width, height/2, facecolor='#00ff00',
                                   transform=handlebox.get_transform())
        handlebox.add_artist(patch1)
        handlebox.add_artist(patch2)
        return patch2

def setup_figure():
    plt.rcParams.update({"font.size": 13})
    f, ax = plt.subplots(1)
    f.set_size_inches(6.2, 5.5)
    f.canvas.layout.width = "6.2in"
    f.canvas.layout.height= "5.5in"
    f.canvas.toolbar_visible = False
    ax.set_yscale("log")
    plt.gcf().subplots_adjust(left=0.135)
    
    return f, ax

    
def plot(ax, data, theory_data, processed_data):
    ax.set_yscale("log")
    ax.set_ylim(*y_limits[data["particle"]])
    ax.fill_between(processed_data.mass_points,
                    np.array(processed_data.mean)-np.array(processed_data.low_2sigma),
                    np.array(processed_data.mean)+np.array(processed_data.high_2sigma), color="#ffff00")
    ax.fill_between(processed_data.mass_points,
                    np.array(processed_data.mean)-np.array(processed_data.low_1sigma),
                    np.array(processed_data.mean)+np.array(processed_data.high_1sigma), color="#00ff00")
    
    exp95, = ax.plot(processed_data.mass_points, processed_data.mean,
                     color="black", linestyle="dashed", linewidth=1, dashes=(5, 5))
    qstar, = ax.plot(theory_data.x, theory_data.y, color="blue", linestyle="dashed", linewidth=1.5, dashes=(6, 4))
    
    pb = float(data["fb"])*1000
    data_x, data_y = [], []
    if len(data["data-limits"]) > 0:
        data_x, data_y = zip(*sorted(data["data-limits"].items()))
        data_y = [y/pb for y in data_y]
    
    obs95, = ax.plot(data_x, data_y, color='black', marker='o', linewidth=1.5, markersize=4)

    yellow_line = mlines.Line2D([], [], color="#ffff00", linewidth=20)
    green_line = mlines.Line2D([], [], color="#ffff00", linewidth=10)

    legend_handles = [qstar, obs95, exp95, PlusMinus()]
    legend_names = ["q* (theory)", "Observed 95% CL upper limit", "Expected 95% CL upper limit", "Expected $\pm1\sigma$ and $\pm2\sigma$"]
    ax.legend(legend_handles, legend_names, loc="lower left", fontsize=10, frameon=False, handler_map={PlusMinus: PlusMinusHandler()})

    ax.text(0.95, 0.95, r"$\sqrt{s}=13\,\mathrm{TeV}$, $37\mathrm{fb}^{-1}$" "\n" r"$|y^{*}|<0.6$", horizontalalignment='right', verticalalignment='top',
            fontsize=12, transform=ax.transAxes)

    ax.set_ylabel(r"$\sigma \times \mathrm{A} \times \mathrm{BR} \, \mathrm{[pb]}$", size=15, horizontalalignment="right", y=1)
    ax.set_xlabel(r"$\mathrm{m}_{\mathrm{q}^{*}}$", size=15, horizontalalignment="right", x=1)
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    ax.tick_params(which="both", direction="in")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    ax.set_title("{} 95% CL limit ({})".format(particle_symbol[data["particle"]], data["workflow"]), pad=12)
