#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2022 Junwen Shen

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
from typing import *
import matplotlib.pyplot as plt
import numpy as np

markers = ['o', '^', 's', 'p', '*', 'h']


def main():
    sizes: List[int] = [32] + [i * 64 for i in range(1, 6)]
    num_threads: List[int] = [i * 2 for i in range(1, 11)]
    sequential_time_records: List[int] = []
    parallel_time_records: Dict[int, Dict[int, List[int]]] = {k: {n: [] for n in num_threads} for k in sizes}
    logs: List[str] = os.listdir('logs')
    for log in logs:
        if log.startswith("sequential"):
            with open(os.path.join('logs', log), 'r') as f:
                sequential_time_records.append(int(f.readline()))
        elif log.startswith("parallel"):
            split: List[str] = log.replace("parallel ", "").replace(".txt", "").split(" ")
            size = int(split[0])
            num_thread = int(split[1])
            with open(os.path.join('logs', log), 'r') as f:
                parallel_time_records[size][num_thread].append(int(f.readlines()[-1]))
    sequential_time_records.sort()
    temp: List[List[int]] = []
    for i in range(len(sizes)):
        t = [sequential_time_records[i]]
        for v in parallel_time_records[sizes[i]].values():
            t.extend(v)
        temp.append(t)
    print(temp)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis([0, 22, 0, 11])
    ax.plot(np.arange(0, 22, 2), np.arange(0, 11, 1) * 2, ls="--", c=".3", label="Linear")
    for i in range(len(sizes)):
        y = []
        for v in parallel_time_records[sizes[i]].values():
            y.extend(v)
        for j in range(len(y)):
            y[j] = sequential_time_records[i] / y[j]
        y = np.array(y)
        ax.plot(num_threads, y, marker=markers[i], label=f"{sizes[i]}M")
    ax.legend()
    plt.xticks(num_threads)
    plt.yticks(np.arange(0, 11, 0.5))
    plt.xlabel('Number of threads')
    plt.ylabel('Speedup')
    plt.title('Speedup of PSRS')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()
    ax.table(cellText=temp, rowLabels=sizes, colLabels=[1] + num_threads, loc='center')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
