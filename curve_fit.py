from __future__ import division
import shelve
from collections import OrderedDict
import copy

from path import path
import yaml
from pymodelfit.fitgui import fit_data
from pymodelfit import LinearModel, LinearInterpolatedModel
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


def get_combined_histogram(s):
    '''
    Generate combined histogram across all netlists in a shelve file.
    '''
    hists = np.array(zip(*[np.histogram(get_deltas_from_netlist(s, netlist),
            bins=50) for netlist in s])[0])
    return np.sum(hists, axis=0)


def get_combined_swap_fraction_to_delta_range_fraction(s):
    #import pudb; pudb.set_trace()
    hist = get_combined_histogram(s)
    count = len(hist)
    plot_data = {'Cummulative swaps count': hist.cumsum() / sum(hist),
        '% of delta range': np.arange(count) / count}
    x_label, y_label = 'Cummulative swaps count', '% of delta range'
    f = LinearInterpolatedModel()
    f.fitData(np.concatenate(([0], plot_data[x_label])), 
            np.concatenate(([0], plot_data[y_label])))
    return f


def get_swap_fraction_to_delta_range_fraction(s, netlist, percent=False):
    deltas = get_deltas_from_netlist(s, netlist, percent=percent)
    hist, bin_edges = np.histogram(deltas, bins=50)
    count = len(hist)
    plot_data = {'Cummulative swaps count': hist.cumsum() / sum(hist),
        '% of delta range': np.arange(count) / count}
    x_label, y_label = 'Cummulative swaps count', '% of delta range'
    f = LinearInterpolatedModel()
    f.fitData(np.concatenate(([0], plot_data[x_label])), 
            np.concatenate(([0], plot_data[y_label])))
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


def get_threshold_parameter_config(input_file):
    s = shelve.open(input_file)
    netlist_info = yaml.load(path('netlist_info.yml').bytes())
    netlist_ranks = OrderedDict([(v[1], i) for i, v in enumerate(netlist_info)])
    netlist_name_ranks = OrderedDict([(v[1].namebase.replace('.', '_'), i)
            for i, v in enumerate(netlist_info)])

    mappings = OrderedDict([(k, get_swap_fraction_to_delta_range_fraction(s, k))
            for k in netlist_name_ranks])
    plt.cla()
    x = np.linspace(0, 1, 1000)
    for k, (f, bin_edges) in mappings.iteritems():
        plt.plot(x, [f(v) for v in x], label=k)
        print '[{:^30}] {:.03g}'.format(k, bin_edges[-1]) # * f(0.5))
    f = get_combined_swap_fraction_to_delta_range_fraction(s)
    swap_fraction_to_delta_range_fraction = f
    plt.plot(x, [f(v) for v in x], label='mean', linewidth=4)
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
    axis.append(plt.plot(block_counts, deltas, label='block vs deltas')[0].axes)
    model = LinearModel()
    model.fitData(block_counts, deltas)
    axis.append(plt.plot(block_counts, [model(v)
            for v in block_counts], label='fitted')[0].axes)
    #for ax in axis:
        #ax.set_yscale('log')
    plt.legend()

    plt.show()

    delta_range_model_data = {'i1d': copy.deepcopy(f.i1d),
            'data': copy.deepcopy(f.data)}
    return delta_range_model_data, model.pardict


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print >> sys.stderr, 'usage: %s <shelve data file>' % sys.argv[0]
        raise SystemExit
    input_file = path(sys.argv[1])
    delta_range_model_data, max_delta_model_params =\
            get_threshold_parameter_config(input_file)

    swap_fraction_to_delta_range_fraction = LinearInterpolatedModel()
    for key, value in delta_range_model_data.iteritems():
        setattr(swap_fraction_to_delta_range_fraction, key, value)
    block_count_to_max_delta = LinearModel(**max_delta_model_params)
