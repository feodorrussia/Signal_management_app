import os
import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import Pool
import pandas as pd


def make_tot_video():
    os.system(
        f"ffmpeg -r 24 -i D:/Edu/Lab/Datas/img/maps/tot/img_%d.png -y D:/Edu/Lab/Datas/vid/filaments_animation_total.avi")


def make_video(frequency):
    os.system(
        f"ffmpeg -r 10 -i D:/Edu/Lab/Datas/img/maps/{frequency}Hz/img_%d.png -vcodec mpeg4 -y D:/Edu/Lab/Datas/vid/filaments_animation_{frequency}Hz.mp4")


def limits_function(frequency):
    ylim = [0, 0]
    for ver in range(11):
        for num in range(28):
            filament = (
                    f"D:/Edu/Lab/Datas/filaments/maps/{frequency}Hz/lines_v{ver}/filament_map_2_{str(frequency).rjust(3, '0')}" +
                    f"Hz_{str(num).rjust(4, '0')}_v0.txt")  #
            if os.path.isfile(filament):
                # print(1)
                with open(filament) as file:
                    text = file.read().strip().split("\n")
                    data = [float(x) for x in text[3:]]
                    if ylim == [0, 0]:
                        ylim[0] = min(data)
                        ylim[1] = max(data)
                    else:
                        ylim[0] = min(min(data), ylim[0])
                        ylim[1] = max(max(data), ylim[1])
    return frequency, ylim


def save_function(frequency, num, limits):
    ylim = limits[frequency]
    curr_fn_data = dataframe.loc[(dataframe['Frequency'] == frequency)]
    x_start_points = curr_fn_data['X_start']
    curr_fn_data = curr_fn_data.loc[curr_fn_data['X_start'] == 0]
    start_num = curr_fn_data.iloc[0].name
    for i in curr_fn_data.index:
        curr_data = curr_fn_data.iloc[i]
        data = list(map(float, curr_data["Values"].split()))

        path = f"D:/Edu/Lab/Datas/img/maps/{frequency}Hz"
        if not os.path.isdir(path):
            os.makedirs(path)

        path_tot = f"D:/Edu/Lab/Datas/img/maps/tot"
        if not os.path.isdir(path_tot):
            os.makedirs(path_tot)

        plt.plot(data)
        plt.figtext(.125, .9, f"fr = {curr_data['Frequency']}Hz v_{(dataframe['X_end'] - dataframe['X_start'])*1e3//4} Num: {num}\n" +
                    f"start point: ({str(round(curr_data['X_start'], 3)).ljust(5, '0')}, {str(round(curr_data['Y_start'], 3)).ljust(5, '0')}), " +
                    f"end point: ({str(round(curr_data['X_end'], 3)).ljust(5, '0')}, {str(round(curr_data['Y_end'], 3)).ljust(5, '0')})")
        plt.grid()
        plt.ylim(ylim[0] * (1 - ylim[0] / abs(ylim[0]) * 0.05), ylim[1] * (1 + ylim[1] / abs(ylim[1]) * 0.05))
        # plt.show()
        # plt.savefig(path + f"/img_{count}.png")
        plt.savefig(path_tot + f"/img_{start_num + i}.png")
        plt.clf()


if __name__ == '__main__':
    path_db = "D:/Edu/Lab/Datas/filaments/maps/"
    name_database = "total_database_v0.csv"
    dataframe = pd.read_csv(path_db+name_database)

    # print(dataframe.head())
    with Pool(10) as p:
        all_limits = dict(p.map(limits_function, list(range(16, 66))))
    save_function(18, 0, all_limits)
        # input_args = [(f, n, all_limits) for f in range(16, 66) for n in range(28)]
        # p.map(save_function, input_args)

    # make_video(freq)

# make_tot_video()
