import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data_generate.data_generation import DataGeneration
from narx_structure.NARX_structure import NARXStructure, NARXParameters


def NARX_filter():
    batch_size = 2
    state_name = ['s', 'f', 'v', 'q']
    hidden_neurons = [3, 3, 11, 3]
    input_TDLs = [5, 5, 1, 4]
    output_TDLs = [1, 1, 1, 1]

    Pas = [NARXParameters(hidden_num=hidden_neurons[i], input_TDL=input_TDLs[i], output_TDL=output_TDLs[i],
                          batch_size=batch_size) for i in range(4)]
    structures = [NARXStructure(Pa=Pas[i]) for i in range(4)]

    with tf.Session() as sess:
        for structure in structures:
            sess.run(structure.init_op)

        [block_neural, block_hemodynamic_state, block_raw_BOLD, block_noisy_interpolated_BOLD] = \
            DataGeneration().data_generation(data_type='block', batch_size=batch_size)

        # data shape: [time_sequence_length, batch_size, data_size]
        block_length = len(block_neural)
        epoch = 0
        while True:
            for index, structure in enumerate(structures):
                out, los = structure.train_time_sequence(input=block_raw_BOLD,
                                                         output=np.reshape(block_hemodynamic_state[:, :, index],
                                                                           newshape=[block_length, batch_size, 1]),
                                                         sess=sess)
                if index == 0:
                    output = out
                    loss = los
                else:
                    output = np.concatenate((output, out), axis=2)
                    loss = np.concatenate((loss, los), axis=1)

            epoch += 1
            loss = np.mean(loss, axis=0)
            print(loss)
            yield [output, block_hemodynamic_state]

NARX_filter()


tdata = []
fig = plt.figure()
subplot_name = ['s', 'f', 'v', 'q']
axs = list()
lines = list()
for i in range(len(subplot_name)):
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xlabel('t')
    ax.set_ylabel(subplot_name[i])
    axs.append(ax)
    if i < 6:
        lines.append([axs[i].plot([], [], lw=2)[0] for _ in range(2)])
    else:
        lines.append(axs[i].plot([], [], lw=2))


def init():
    for ax, line in zip(axs, lines):
        ax.set_ylim(-0.1, 0.1)
        ax.set_xlim(0, 100)
        ax.grid()
        for l in line:
            l.set_data([], [])

    return lines,


def run(data):
    state = data
    # Data Pre-processing
    # state
    state = np.array(state)

    [line_num, time_length, batch_size, varible_num] = np.shape(state)
    step_size = 0.1

    s = state[:, :, 0, 0]
    f = state[:, :, 0, 1]
    v = state[:, :, 0, 2]
    q = state[:, :, 0, 3]
    del state

    datas = [s, f, v, q]

    # Neural
    for ax, line, data, type in zip(axs, lines, datas, subplot_name):
        # Axis Range
        min_value = np.min(data)
        min_value -= min_value / 10

        max_value = np.max(data)
        max_value += max_value / 10

        ax.set_ylim(min_value, max_value)

        ax.set_xlim(0, time_length * step_size)
        tdata = np.arange(0, time_length) * step_size
        for l, d in zip(line, data):
            l.set_data(tdata, d[0:len(tdata)])
        ax.figure.canvas.draw()
    return lines,

ani = animation.FuncAnimation(fig, run, NARX_filter, blit=False,
                              repeat=False, init_func=init)
plt.show()