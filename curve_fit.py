from __future__ import division
import shelve
from collections import OrderedDict

from path import path
import yaml
from pymodelfit.fitgui import fit_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate.interpolate import interp1d


def get_combined_data(s, netlist):
    netlist_data = s[netlist]
    netlist_combined_data = dict([(k, np.concatenate([netlist_data[i][k]
            for i in range(len(netlist_data))]))
                    for k in netlist_data[0].keys()])
    non_zero = np.where(netlist_combined_data['parent_mean'] > 0)
    return dict([(k, v[non_zero])
        for k, v in netlist_combined_data.iteritems()])


def get_deltas(netlist_data, percent=False):
    parent_mean, child_mean = (netlist_data[k]
            for k in ('parent_mean', 'child_mean'))
    if percent:
        deltas = (parent_mean - child_mean) / parent_mean
    else:
        deltas = parent_mean - child_mean
    return np.array(sorted(deltas))


def get_model(s, netlist):
    netlist_combined_data = get_combined_data(s, netlist)
    deltas = get_deltas(netlist_combined_data)
    count = len(deltas)
    hist, bin_edges = np.histogram(deltas)
    model = fit_data(range(len(hist)), hist / count)
    return model, netlist_combined_data


def plot_netlist_cdf(s, netlist):
    data = get_combined_data(s, netlist)
    deltas = get_deltas(data)
    hist, bin_edges = np.histogram(deltas, bins=50)
    print max(hist), min(hist), hist[0], hist[-1]
    print hist.cumsum()
    count = len(hist)
    plot_data = {'Cummulative swaps count': hist.cumsum() / sum(hist),
        '% of delta range': np.arange(count) / count}
    x_label, y_label = 'Cummulative swaps count', '% of delta range'
    plt.plot(plot_data[x_label], plot_data[y_label], label=netlist)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def get_swap_fraction_to_delta_range_fraction(s, netlist, percent=False):
    deltas = get_deltas_from_netlist(s, netlist, percent=percent)
    hist, bin_edges = np.histogram(deltas, bins=50)
    count = len(hist)
    plot_data = {'Cummulative swaps count': hist.cumsum() / sum(hist),
        '% of delta range': np.arange(count) / count}
    x_label, y_label = 'Cummulative swaps count', '% of delta range'
    f = interp1d(plot_data[x_label], plot_data[y_label])
    return f, bin_edges


def get_deltas_from_netlist(s, netlist, percent=False):
    data = get_combined_data(s, netlist)
    return get_deltas(data, percent=percent)


class LinearFunction(object):
    def __init__(self, m=1, b=0):
        self.m =m
        self.b = b

    def __call__(self, x):
        """
        Compute the output of the function at the input value(s) `x`.

        :param x: The input values as an ndarray or a scalar input value.

        :returns: The output of the function - shape will match `x`
        """
        arr = np.array(x,copy=False,dtype=float)
        res = self.m * arr + self.b
        return res.reshape(arr.shape)
 


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print >> sys.stderr, 'usage: %s <shelve data file>' % sys.argv[0]
        raise SystemExit
    input_file = path(sys.argv[1])
    
    s = shelve.open(input_file)
    netlist_info = yaml.load(path('netlist_info.yml').bytes())
    netlist_ranks = OrderedDict([(v[1], i) for i, v in enumerate(netlist_info)])
    netlist_name_ranks = OrderedDict([(v[1].namebase.replace('.', '_'), i)
            for i, v in enumerate(netlist_info)])

    mappings = OrderedDict([(k, get_swap_fraction_to_delta_range_fraction(s, k))
            for k in netlist_name_ranks])
    plt.cla()
    x = np.linspace(0.2, 1, 1000)
    for k, (f, bin_edges) in mappings.iteritems():
        plt.plot(x, [f(v) for v in x], label=k)
        print '[{:^30}] {:.03g}'.format(k, bin_edges[-1]) # * f(0.5))
    plt.plot(x, [np.array([f(v)
            for (f, bin_edges) in mappings.itervalues()]).mean()
                    for v in x], label='mean', linewidth=4)
    x_label, y_label = 'Cummulative swaps count', '% of delta range'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

    plt.figure()

    ordered_deltas = OrderedDict([(netlist,
            get_deltas_from_netlist(s, netlist.namebase.replace('.', '_')).max())
                    for netlist in netlist_ranks])
    deltas = np.array(ordered_deltas.values())
    normalized_deltas = deltas / deltas.max()

    block_counts = np.array([v[0]['block_count'] for v in netlist_info])
    normalized_block_counts = block_counts / block_counts.max()

    net_counts = np.array([v[0]['net_count'] for v in netlist_info])
    normalized_net_counts = net_counts / net_counts.max()

    axis =[]
    axis.append(plt.plot(deltas, block_counts, label='block vs deltas')[0].axes)
    path('block_vs_deltas.pickled').pickle_dump((deltas, block_counts))
    #for ax in axis:
        #ax.set_yscale('log')
    plt.legend()

    plt.show()
