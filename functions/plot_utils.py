import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid", {"axes.facecolor": ".95"})


def plot_sgd(loss_path, step_test, loss_train_svm, loss_test_svm, acc_train_svm, acc_test_svm, output_train_svm,
             output_test_svm, loss_train_nn, loss_test_nn, acc_train_nn, acc_test_nn, output_train_nn, output_test_nn):
    # plot at most 20 points for NN
    gap = np.ceil(len(step_test) / 20).astype(int)
    step_nn = step_test[::gap]
    step_test = np.unique(step_test)
    step_nn = np.unique(step_nn)
    nn_ids = np.where(np.in1d(step_test, step_nn))[0]

    # plot
    # plt.figure(figsize=(22, 5), dpi=500)
    plt.figure(figsize=(30, 7), dpi=500)
    title_size = 40
    plt.rcParams.update({'font.size': 22})
    plt.rc('legend', fontsize=22)
    plt.rc('figure', titlesize=title_size)  # fontsize of the figure title

    marker_style = dict(marker='.', linestyle='None', markersize=10, markerfacecolor='None')
    # marker_style = dict(marker='*', linestyle='None', markersize=4)

    ax = plt.subplot(1, 4, 1)
    for i in range(5):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(step_test, output_train_svm[:, i], linestyle='-', color=color, linewidth=1, label='SVM (infinite NN)')
        plt.plot(step_nn, output_train_nn[:, i][nn_ids], **marker_style, color=color, label='NN')
        if i == 0:
            plt.legend(loc=2)
            leg = ax.get_legend()
            for handles in leg.legendHandles:
                handles.set_color('black')
    ax.set_xscale("log", base=10)
    plt.xlabel('step', fontsize=title_size)
    plt.title('Train outputs', fontsize=title_size)

    ax = plt.subplot(1, 4, 2)
    for i in range(5):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(step_test, output_test_svm[:, i], linestyle='-', color=color, linewidth=1, label='SVM (infinite NN)')
        plt.plot(step_nn, output_test_nn[:, i][nn_ids], **marker_style, color=color, label='NN')
        if i == 0:
            plt.legend(loc=2)
            leg = ax.get_legend()
            for handles in leg.legendHandles:
                handles.set_color('black')
    ax.set_xscale("log", base=10)
    plt.xlabel('step', fontsize=title_size)
    plt.title('Test outputs', fontsize=title_size)

    ax = plt.subplot(1, 4, 3)
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(step_test, loss_train_svm, linestyle='-', color=color, linewidth=1, label='SVM train')
    plt.plot(step_nn, loss_train_nn[nn_ids], **marker_style, color=color, label='NN train')
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(step_test, loss_test_svm, linestyle='-', color=color, linewidth=1, label='SVM test')
    plt.plot(step_nn, loss_test_nn[nn_ids], **marker_style, color=color, label='NN test')
    ax.set_xscale("log", base=10)
    plt.legend()
    leg = ax.get_legend()
    for handles in leg.legendHandles:
        handles.set_color('black')
    plt.xlabel('step', fontsize=title_size)
    plt.title('Hinge loss', fontsize=title_size)
    plt.tight_layout()

    ax = plt.subplot(1, 4, 4)
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(step_test, acc_train_svm, linestyle='-', color=color, linewidth=1, label='SVM train')
    plt.plot(step_nn, acc_train_nn[nn_ids], **marker_style, color=color, label='NN train')
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(step_test, acc_test_svm, linestyle='-', color=color, linewidth=1, label='SVM test')
    plt.plot(step_nn, acc_test_nn[nn_ids], **marker_style, color=color, label='NN test')
    ax.set_xscale("log", base=10)
    plt.legend()
    leg = ax.get_legend()
    for handles in leg.legendHandles:
        handles.set_color('black')
    plt.xlabel('step', fontsize=title_size)
    plt.title('Accuracy', fontsize=title_size)
    plt.tight_layout()

    plt.savefig(loss_path)
    plt.close()
