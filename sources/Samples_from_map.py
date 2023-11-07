import os
import random
from enum import Enum
from multiprocessing import Pool, freeze_support
from functools import partial

import math as m
import numpy as np
from scipy import fft
import scipy.interpolate as sc_i
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Showing map with Bresenham Line plot
def show_Map_with_Line(c_map, line):
    sns.set()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=80)
    sns.heatmap(ax=ax[0], data=c_map)
    ax[0].plot(line[1], line[0], color="white", linewidth=2)
    sns.lineplot(ax=ax[1], data=bresenham_line(c_map, line[0][0], line[1][0], line[0][1], line[1][1]))
    plt.show()


def show_Signal(vals, t=4 / 5, add_data=None):
    if add_data is None:
        add_data = [29, 4, 10, [0, 0], [0, 0], "h_mode"]

    plt.figure(1)
    plt.plot(np.linspace(0, vals.shape[-1], vals.shape[-1]), vals)
    plt.figtext(.125, .9, f"fr = {add_data[0]}Hz Rate: {add_data[1]} Speed: {add_data[2]} Noise: {add_data[-1]}\n" +
                f"start point: ({str(round(add_data[3][0], 3)).ljust(5, '0')}, {str(round(add_data[3][1], 3)).ljust(5, '0')}), " +
                f"end point: ({str(round(add_data[4][0], 3)).ljust(5, '0')}, {str(round(add_data[4][1], 3)).ljust(5, '0')})")
    plt.grid()
    # plt.show(block=False)
    # plt.pause(t)

    path_samples = f"D:/Edu/Lab/Datas/img/samples"
    if not os.path.isdir(path_samples):
        os.makedirs(path_samples)

    plt.savefig(path_samples + f"/img_{add_data[0]}Hz_{add_data[1]}MHz_{add_data[2]}kps_{add_data[5]}.png")
    plt.clf()
    plt.close(1)


# Bresenham Line
def bresenham_line(matrix, i1, j1, i2, j2):
    result = []
    dx = abs(j2 - j1)
    dy = abs(i2 - i1)
    sx = 1 if j1 < j2 else -1
    sy = 1 if i1 < i2 else -1
    err = dx - dy
    while True:
        result.append(matrix[i1][j1])
        if j1 == j2 and i1 == i2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            j1 += sx
        if e2 < dx:
            err += dx
            i1 += sy
    return result


class NoiseMode(Enum):
    low = 1
    normal = 2
    high = 3
    mix = 4
    random = 0


def gen_Noise(data: np.array, mode: NoiseMode):
    mu_re, sigma_re = random.random() * 2 + 1, random.random() * 3 + 2
    real_part = np.random.default_rng().normal(loc=mu_re, scale=sigma_re, size=(len(data) * 50,))
    imag_part = np.random.default_rng().normal(loc=mu_re, scale=sigma_re, size=(len(data) * 50,))

    frequencies_values_gauss = real_part + 1j * imag_part
    noise_data = fft.ifft(frequencies_values_gauss)[1:]

    start_ind = random.randint(0, noise_data.shape[-1] - data.shape[-1] - 1)
    noise_data = noise_data[start_ind:start_ind + data.shape[-1]]

    d = 1e-1
    data_noise_cof = (max(data) - min(data)) / (max(noise_data) - min(noise_data))
    if mode == NoiseMode.mix:
        n_segment = random.randint(5, len(data) // 5)
        for i_segment in range(n_segment):
            d_ = random.randrange(2, 16) / 2 * 1e-1 + (random.random() - 0.5) * 3 * 1e-1
            noise_data[
            i_segment * len(data) // n_segment:(i_segment + 1) * len(data) // n_segment] *= d_ * data_noise_cof
        return noise_data
    if mode == NoiseMode.low:
        d = 2 * 1e-1
    if mode == NoiseMode.normal:
        d = 3 * 1e-1
    if mode == NoiseMode.high:
        d = 8 * 1e-1
    if mode == NoiseMode.random:
        d = 2 * 1e-1 + (random.random() - 0.5) * 3 * 1e-1

    return noise_data * data_noise_cof * d


def increasing_Rate_quadratic(values: np.array, new_n):
    f = sc_i.interp1d(np.linspace(0, len(values), len(values)), values, kind="quadratic")
    return f(np.linspace(0, len(values), int(new_n)))


def signal_augmentation(values, add_data, base_rate, new_speed, dxdy):

    result = []
    new_number = dxdy * len(values) * base_rate * 1e3 / new_speed
    new_rate_values = increasing_Rate_quadratic(values, new_number)

    for v_i in range(1):
        noise_values = new_rate_values + gen_Noise(new_rate_values, mode=NoiseMode.low)
        # show_Signal(noise_values, t=1 / 5)
        result.append(
            [add_data[0], base_rate, new_speed, [add_data[3][0], add_data[2][0]], [add_data[3][1], add_data[2][1]],
             f"low_mode_{v_i}", noise_values.real])
    for v_i in range(1):
        noise_values = new_rate_values + gen_Noise(new_rate_values, mode=NoiseMode.random)
        # show_Signal(noise_values, t=1 / 5)
        result.append(
            [add_data[0], base_rate, new_speed, [add_data[3][0], add_data[2][0]], [add_data[3][1], add_data[2][1]],
             f"rand_mode_{v_i}", noise_values.real])  # ' '.join(list(map(str, noise_values.real)))
    for v_i in range(1):
        noise_values = new_rate_values + gen_Noise(new_rate_values, mode=NoiseMode.high)
        # show_Signal(noise_values, t=1 / 5)
        result.append(
            [add_data[0], base_rate, new_speed, [add_data[3][0], add_data[2][0]], [add_data[3][1], add_data[2][1]],
             f"high_mode_{v_i}", noise_values.real])
    for v_i in range(3):
        noise_values = new_rate_values + gen_Noise(new_rate_values, mode=NoiseMode.mix)
        # show_Signal(noise_values, t=1 / 5)
        result.append(
            [add_data[0], base_rate, new_speed, [add_data[3][0], add_data[2][0]], [add_data[3][1], add_data[2][1]],
             f"mix_mode_{v_i}", noise_values.real])
    return result


def check_Line_toFilament(values: np.array, add_data=None, auto_save=False):
    if add_data is None:
        add_data = [29, 0, [0, 0], [0, 0]]

    if auto_save:
        zero_level = values[0]
        min_y, max_y = min(values), max(values)
        n_data = list(filter(lambda el: el != min_y and el != max_y, values))
        save_fl = "y"
        if len(n_data) != 0:
            min2_y, max2_y = min(n_data), max(n_data)
            if (abs((min_y - max_y) / zero_level) < 1e-1 or abs((min_y - zero_level) / (min2_y - zero_level)) > 2
                    or abs((max_y - zero_level) / (max2_y - zero_level)) > 2):
                print(abs((min_y - max_y) / zero_level), "< 0.1", abs((min_y - zero_level) / (min2_y - zero_level)),
                      "> 2", abs((max_y - zero_level) / (max2_y - zero_level)), "> 2")
                show_Signal(values)
                save_fl = input(
                    f"{add_data[0]}Hz_v{int((add_data[3][1] - add_data[3][0]) / dx_map) + 5}_{add_data[1]}. To don't save this signal, press [n]: ")
    else:
        show_Signal(values)
        save_fl = input(
            f"{add_data[0]}Hz_v{int((add_data[3][1] - add_data[3][0]) / dx_map) + 5}_{add_data[1]}. To don't save this signal, press [n]: ")

    if not (add_data is None or save_fl.strip().lower() in ["n", "т"]):
        return [True, values, add_data]
    return [False, values, add_data]


if __name__ == '__main__':
    freeze_support()
    # Set the number of files
    n = 6000

    # Set path to dir with maps files
    path_to_map = "D:/Edu/Lab/Datas/maps/map_2/"
    map_name = "map_2"

    # Set scale coefficients of map
    dx_map, dy_map = 4 * 1e-3, 4 * 1e-3

    # Set path to dir for save filaments
    path_to_filaments = "D:/Edu/Lab/Datas/filaments/maps/"

    path_db = "D:/Edu/Lab/Datas/filaments/maps/"
    name_database = "db_30Hz_s15_f_10_10_10_30.csv"

    # Making an array of positions with id
    with open(path_to_map + "positions.txt", "r") as file_of_positions:
        lines = file_of_positions.readlines()[2:]
        lines = [line.replace("\n", "").split(" ") for line in lines]
        lines = [[round(float(line[0])), float(line[1]), float(line[2])] for line in lines]
        positions = pd.DataFrame(lines, columns=["id", "r", "z"])

    # Create list with ids
    ids = []
    for i in range(n):
        name = path_to_map + "globus_" + str(i).rjust(4, "0") + ".txt"
        with open(name) as file_of_values:
            lines = file_of_values.readlines()[2:]
            lines = [line.split(" ")[:-1] for line in lines]
            lines = [[float(line[0]), float(line[1])] for line in lines]
        ids.append(lines)

    # Making plot of the positions
    x = sorted(list(set(positions.r)))
    y = sorted(list(set(positions.z)))

    # Creating r variable for the matrix
    width = len(x)
    height = len(y)
    curr_map = np.zeros((height, width))

    x_, y_ = np.meshgrid(x, y)

    frequency = 29

    order_num = 0
    for r, z in zip(np.hstack(x_), np.hstack(y_)):
        id_pos = positions.loc[(positions["r"] == r) & (positions["z"] == z), 'id'].iloc[0]
        for ids_line in ids[id_pos]:
            if abs(ids_line[0] - frequency * 10 ** 9) < 10:
                curr_map[order_num // width][order_num % width] = ids_line[1]
        order_num += 1

    curr_map = curr_map - (np.abs(curr_map) > 1) * curr_map
    curr_map = np.round(curr_map, 4)

    i = 5
    x_coord_top = int(28 + 0.19 * (65 - frequency)) + i
    coord_dx = 2
    line_coord = [[0, 99], [x_coord_top, x_coord_top + coord_dx]]

    board_f = True

    board_f, values_, add_data_ = check_Line_toFilament(
        values=np.array(bresenham_line(curr_map, line_coord[0][0], line_coord[1][0], line_coord[0][1],
                                       line_coord[1][1])),
        add_data=[frequency, i, [line_coord[0][0] * dy_map, line_coord[0][1] * dy_map],
                  [line_coord[1][0] * dx_map, line_coord[1][1] * dx_map]],
        auto_save=board_f)

    n_v = 16  # [1, 50], больше вблизи 16
    rate = 4  # 10
    base_s = dx_map * rate * 1e3  # from formula of speed: v = l/t = l*w

    speeds = np.logspace(0, 2, 10)
    speeds = speeds[(speeds < 50)]  # & (speeds > 16)
    speeds = np.append(speeds, [14, 14.8, 15.5, 16, 16.5, 17.2, 18, 50])
    speeds = speeds.round(2)
    speeds.sort()
    print(speeds)

    for speed in speeds:
        add_filaments = signal_augmentation(values_, add_data_, rate, speed, dx_map)

        for filament in add_filaments:
            show_Signal(filament[-1], add_data=filament[:-1])

    #
    #
