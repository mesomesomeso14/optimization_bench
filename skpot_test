import numpy as np
from skopt import gp_minimize, gbrt_minimize, forest_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_evaluations, plot_objective, plot_regret
import matplotlib.pyplot as plt
import os


def easom(a, b):
    return -1 * np.cos(a) * np.cos(b) * np.exp(-1 * (a - np.pi) ** 2 - (b - np.pi) ** 2)


def ackley(a, b):
    return 20 - 20 * np.exp(-0.2 * np.sqrt((a ** 2 + b ** 2) / 2)) + np.e - np.exp(
        (np.cos(2 * np.pi * a) + np.cos(2 * np.pi * b)) / 2)


def bukin_6(a, b):
    return 100 * np.sqrt(np.abs(b - 0.01 * a ** 2)) + 0.01 * np.abs(a + 10)


def levi_13(a, b):
    return (np.sin(3 * np.pi * a)) ** 2 + ((a - 1) ** 2) * (1 + (np.sin(3 * np.pi * b)) ** 2) + ((b - 1) ** 2) * (
            1 + (np.sin(2 * np.pi * b)) ** 2)


def schaffer_4(a, b):
    return 0.5 + ((np.cos(np.sin(np.abs(a ** 2 - b ** 2)))) ** 2 - 0.5) / (1 + 0.001 * (a ** 2 + b ** 2)) ** 2


def five_well(a, b):
    A = 1 / (1 + 0.05 * (a ** 2 + (b - 10) ** 2))
    B = 1 / (1 + 0.05 * ((a - 10) ** 2 + b ** 2))
    C = 1.5 / (1 + 0.03 * ((a + 10) ** 2 + b ** 2))
    D = 2 / (1 + 0.05 * ((a - 5) ** 2 + (b + 10) ** 2))
    E = 1 / (1 + 0.1 * ((a + 5) ** 2 + (b + 10) ** 2))
    F = 0.0001 * (a ** 2 + b ** 2) ** 1.2
    return (1 - A - B - C - D - E) * (1 + F)


def three_hump_camel(a, b):
    return 2 * a ** 2 - 1.05 * a ** 4 + (a ** 6) / 6 + a * b + b ** 2


def six_hump_camel(a, b):
    return (4 - 2.1 * a ** 2 + (a ** 4) / 3) * a ** 2 + a * b + 4 * (b ** 2 - 1) * b ** 2


def eggholder(a, b):
    return -1 * (b + 47) * np.sin(np.sqrt(np.abs(b + a / 2 + 47))) - a * np.sin(np.sqrt(np.abs(a - (b + 47))))


ncall, nrand = 500, 100
easom_area = [-100, -100, 100, 100]
ackley_area = [-32.768, -32.768, 32.768, 32.768]
bukin_6_area = [-15, -3, -5, 3]
levi_13_area = [-10, -10, 10, 10]
schaffer_4_area = [-100, -100, 100, 100]
five_well_area = [-20, -20, 20, 20]
three_hump_camel_area = [-5, -5, 5, 5]
six_hump_camel_area = [-3, -2, 3, 2]
eggholder_area = [-512, -512, 512, 512]

funclist = [eggholder, three_hump_camel, six_hump_camel, easom,
            ackley, bukin_6, levi_13, schaffer_4, five_well]
arealist = [eggholder_area, three_hump_camel_area, six_hump_camel_area, easom_area,
            ackley_area, bukin_6_area, levi_13_area, schaffer_4_area, five_well_area]

for func, area in zip(funclist, arealist):

    f = './181221/' + func.__name__
    print(f)
    if not os.path.isdir(f):
        os.mkdir(f)
    [vmin1, vmin2, vmax1, vmax2] = area

    space = [Real(vmin1, vmax1, name='x1'), Real(vmin2, vmax2, name='x2')]


    @use_named_args(space)
    def objective(**params):
        x1 = params['x1']
        x2 = params['x2']
        score = func(x1, x2)
        return score


    for acq in ['gp_hedge', 'EI', 'LCB', 'PI']:
        result = gbrt_minimize(objective, space, n_calls=ncall, acq_func=acq, n_random_starts=nrand,
                               random_state=220, verbose=True, n_jobs=1)
        value = np.array(result.x_iters)
        plt.figure(figsize=(10, 8))
        plt.scatter(value[:, 0], value[:, 1], c=result.func_vals, cmap='bwr')
        plt.colorbar()
        plt.grid(linestyle=':')
        plt.xlim([vmin1, vmax1])
        plt.ylim([vmin2, vmax2])
        plt.savefig(f + '/gbrt_plot_result_%s.png' % acq)
        plt.close()

        print('     Plot_convergence ...')
        plot_convergence(result)
        plt.savefig(f + '/gbrt_convergence_%s.png' % acq)
        plt.close()

        print('     Plot_objective ...')
        plot_objective(result)
        plt.savefig(f + '/gbrt_objective_%s.png' % acq)
        plt.close()

        print('     plot_evaluations ...')
        plot_evaluations(result)
        plt.savefig(f + '/gbrt_evaluations_%s.png' % acq)
        plt.close()

        print('     plot_regret ...')
        plot_regret(result)
        plt.savefig(f + '/gbrt_regret_%s.png' % acq)
        plt.close()

        result = forest_minimize(objective, space, n_calls=ncall, acq_func=acq, n_random_starts=nrand,
                                 random_state=220, verbose=True, n_jobs=1)
        value = np.array(result.x_iters)
        plt.figure(figsize=(10, 8))
        plt.scatter(value[:, 0], value[:, 1], c=result.func_vals, cmap='bwr')
        plt.colorbar()
        plt.grid(linestyle=':')
        plt.xlim([vmin1, vmax1])
        plt.ylim([vmin2, vmax2])
        plt.savefig(f + '/forest_plot_result_%s.png' % acq)
        plt.close()

        print('     Plot_convergence ...')
        plot_convergence(result)
        plt.savefig(f + '/forest_convergence_%s.png' % acq)
        plt.close()

        print('     Plot_objective ...')
        plot_objective(result)
        plt.savefig(f + '/forest_objective_%s.png' % acq)
        plt.close()

        print('     plot_evaluations ...')
        plot_evaluations(result)
        plt.savefig(f + '/forest_evaluations_%s.png' % acq)
        plt.close()

        print('     plot_regret ...')
        plot_regret(result)
        plt.savefig(f + '/forest_regret_%s.png' % acq)
        plt.close()

        result = gp_minimize(objective, space, n_calls=ncall, acq_func=acq, n_random_starts=nrand,
                             random_state=220, verbose=True, n_jobs=1)
        value = np.array(result.x_iters)
        plt.figure(figsize=(10, 8))
        plt.scatter(value[:, 0], value[:, 1], c=result.func_vals, cmap='bwr')
        plt.colorbar()
        plt.grid(linestyle=':')
        plt.xlim([vmin1, vmax1])
        plt.ylim([vmin2, vmax2])
        plt.savefig(f + '/gp_plot_result_%s.png' % acq)
        plt.close()

        print('     Plot_convergence ...')
        plot_convergence(result)
        plt.savefig(f + '/gp_convergence_%s.png' % acq)
        plt.close()

        print('     Plot_objective ...')
        plot_objective(result)
        plt.savefig(f + '/gp_objective_%s.png' % acq)
        plt.close()

        print('     plot_evaluations ...')
        plot_evaluations(result)
        plt.savefig(f + '/gp_evaluations_%s.png' % acq)
        plt.close()

        print('     plot_regret ...')
        plot_regret(result)
        plt.savefig(f + '/gp_regret_%s.png' % acq)
        plt.close()
