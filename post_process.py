import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


def read_3d_data(frame: int):
    data = np.genfromtxt(f"data/{frame}.csv", delimiter=",")
    if len(data.shape) == 1:
        data = data.reshape((-1, data.shape[0]))
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    return x, y, z


def plot_3d_position(frame: int):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    x, y, z = read_3d_data(frame)
    ax.plot(x, y, z, ".")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    plt.show()


def make_3d_anime():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    x, y, z = read_3d_data(0)
    (line,) = ax.plot(x, y, z, ".")

    def update(frame: int):
        x, y, z = read_3d_data(frame)
        line.set_data_3d(x, y, z)
        return (line,)

    files = os.listdir("data")
    anime = animation.FuncAnimation(
        fig=fig, func=update, frames=tqdm(range(len(files)))
    )
    anime.save("anime.mp4")

def read_2d_data(frame: int):
    data = np.genfromtxt(f"data/{frame}.csv", delimiter=",")
    if len(data.shape) == 1:
        data = data.reshape((-1, data.shape[0]))
    x = data[:, 0]
    y = data[:, 1]
    return x, y

def make_2d_anime():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x, y = read_2d_data(0)
    (line,) = ax.plot(x, y, ".")

    def update(frame: int):
        x, y = read_2d_data(frame)
        line.set_xdata(x)
        line.set_ydata(y)
        return (line,)

    files = os.listdir("data")
    anime = animation.FuncAnimation(
        fig=fig, func=update, interval=42, frames=tqdm(range(len(files)))
    )
    anime.save("anime.mp4")


if __name__ == "__main__":
    # make_3d_anime()
    make_2d_anime()
