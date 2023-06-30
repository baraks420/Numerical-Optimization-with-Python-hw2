import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_result(x_list):
    lines = create_lines()
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"linear")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.plot(lines[0][0], lines[0][1], 'r', linestyle='dotted', label="y= -x+1")
    plt.plot(lines[1][0], lines[1][1], 'g', linestyle='dotted', label="y=0")
    plt.plot(lines[2][0], lines[2][1], 'b', linestyle='dotted', label="y=1")
    plt.plot(lines[3][0], lines[3][1], 'c', linestyle='dotted', label="x=2")
    x, y = create_fill()
    plt.fill(x, y, 'y', label="feasible region")
    xs = []
    ys = []
    for point in x_list:
        xs.append(point[0])
        ys.append(point[1])
    plt.plot(xs, ys, 'black', linewidth=3, label="path")
    plt.scatter(x_list[-1][0], x_list[-1][1], s=100, c='r', marker='*', label='final point')
    plt.legend()
    plt.show()


def plot_result_3d(x_list,angle):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = [1, 0, 0]
    y = [0, 1, 0]
    z = [0, 0, 1]

    # Plot the triangle
    ax.plot_trisurf(x, y, z, color='red', alpha=0.5)
    ax.set_title(f"quadratic")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    xs = []
    ys = []
    zs = []
    for point in x_list:
        xs.append(point[0])
        ys.append(point[1])
        zs.append(point[2])
    ax.plot(xs,ys,zs ,label = "path")
    ax.scatter(x_list[-1][0], x_list[-1][1],x_list[-1][2], s=100, c='green', marker='*', label='final point')

    plt.legend()
    ax.view_init(angle[0],angle[1])
    plt.show()


def plot_value_iterations(objectives, name_of_function=""):
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"objective per iteration - {name_of_function}")
    ax.set_xlabel('iteration number')
    ax.set_ylabel('f(x)')
    _ = ax.plot(range(len(objectives)), objectives)
    plt.legend()
    plt.show()


def create_lines():
    x = np.linspace(-1, 3, 1000)
    y1 = -x + 1
    y2 = x * 0
    y3 = x * 0 + 1
    y4 = np.linspace(-2, 2, 1000)
    x4 = y4 * 0 + 2
    return (x, y1), (x, y2), (x, y3), (x4, y4)


def create_fill():
    x1 = np.linspace(0, 2, 10)
    y1 = np.linspace(1, 1, 10)
    x2 = np.linspace(2, 2, 10)
    y2 = np.linspace(1, 0, 10)
    x3 = np.linspace(2, 1, 10)
    y3 = np.linspace(0, 0, 10)
    x4 = np.linspace(1, 0, 10)
    y4 = np.linspace(0, 1, 10)
    return np.concatenate((x1, x2, x3, x4)), np.concatenate((y1, y2, y3, y4)),
