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


def show_Signal(vals, t=4 / 5):
    plt.figure(1)
    plt.plot(np.linspace(0, vals.shape[-1], vals.shape[-1]), vals)
    plt.grid()
    plt.show(block=False)
    plt.pause(t)
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


def increasing_Rate_quadratic(values: np.array, rate, new_rate):
    f = sc_i.interp1d(np.linspace(0, len(values), len(values)), values, kind="quadratic")
    return f(np.linspace(0, len(values), int(new_rate / rate * len(values))))


def signal_augmentation(values, add_data, r_i):
    base_rate = 5
    base_speed = 27.65
    rate_speed = {base_rate: base_speed}  # for the future

    result = []
    # for r_i in range(1, 51):
    new_rate = base_rate * r_i / 2
    new_speed = base_speed * m.log(r_i / 2 * m.e)
    new_rate_values = increasing_Rate_quadratic(values, base_rate, new_rate)

    for v_i in range(10):
        noise_values = new_rate_values + gen_Noise(new_rate_values, mode=NoiseMode.low)
        # show_Signal(noise_values, t=1 / 5)
        result.append([add_data[0], new_rate, new_speed, add_data[3][0], add_data[2][0], add_data[3][1], add_data[2][1],
                       ' '.join(list(map(str, noise_values.real)))])
    for v_i in range(10):
        noise_values = new_rate_values + gen_Noise(new_rate_values, mode=NoiseMode.random)
        # show_Signal(noise_values, t=1 / 5)
        result.append([add_data[0], new_rate, new_speed, add_data[3][0], add_data[2][0], add_data[3][1], add_data[2][1],
                       ' '.join(list(map(str, noise_values.real)))])
    for v_i in range(10):
        noise_values = new_rate_values + gen_Noise(new_rate_values, mode=NoiseMode.high)
        # show_Signal(noise_values, t=1 / 5)
        result.append([add_data[0], new_rate, new_speed, add_data[3][0], add_data[2][0], add_data[3][1], add_data[2][1],
                       ' '.join(list(map(str, noise_values.real)))])
    for v_i in range(30):
        noise_values = new_rate_values + gen_Noise(new_rate_values, mode=NoiseMode.mix)
        # show_Signal(noise_values, t=1 / 5)
        result.append([add_data[0], new_rate, new_speed, add_data[3][0], add_data[2][0], add_data[3][1], add_data[2][1],
                       ' '.join(list(map(str, noise_values.real)))])
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

    db_columns = ['Frequency', 'Rate', 'Speed', 'X_start', 'Y_start', 'X_end', 'Y_end', 'Values']

    data = []
    df = pd.DataFrame(data, columns=db_columns)
    df.to_csv(path_to_filaments + name_database)

    for frequency in range(29, 30):
        order_num = 0
        for r, z in zip(np.hstack(x_), np.hstack(y_)):
            id_pos = positions.loc[(positions["r"] == r) & (positions["z"] == z), 'id'].iloc[0]
            for ids_line in ids[id_pos]:
                if abs(ids_line[0] - frequency * 10 ** 9) < 10:
                    curr_map[order_num // width][order_num % width] = ids_line[1]
            order_num += 1

        curr_map = curr_map - (np.abs(curr_map) > 1) * curr_map
        curr_map = np.round(curr_map, 4)

        # Make r line
        x_coord_top = int(28 + 0.19 * (65 - frequency))
        for coord_dx in range(-5, 6):
            board_f = False
            version = 5 + coord_dx
            line_coord = [[0, 99], [x_coord_top, x_coord_top + coord_dx]]

            delta = 3
            h = 1

            k = width - max(line_coord[1]) - delta * h - 1
            for i in range(k):
                line_coord[1][1] += h
                line_coord[1][0] += h
                # show_Map_with_Line(curr_map, line_coord)

                board_f, values_, add_data_ = check_Line_toFilament(
                    values=np.array(bresenham_line(curr_map, line_coord[0][0], line_coord[1][0], line_coord[0][1],
                                                   line_coord[1][1])),
                    add_data=[frequency, i, [line_coord[0][0] * dy_map, line_coord[0][1] * dy_map],
                              [line_coord[1][0] * dx_map, line_coord[1][1] * dx_map]],
                    auto_save=board_f)

                if board_f:
                    if not os.path.isdir(path_db):
                        os.makedirs(path_db)
                    with Pool(10) as p:
                        p_signal_augmentation = partial(signal_augmentation, values_, add_data_)
                        add_filaments = p.map(p_signal_augmentation, list(range(1, 16)))

                    data_add = []
                    for el in add_filaments:
                        data_add += el

                    dataframe_add = pd.DataFrame(data_add, columns=db_columns)
                    dataframe_add.to_csv(path_db + name_database, mode='a', header=False)
    #
    #
