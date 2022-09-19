/*
 * Copyright (C) 2022 Junwen Shen
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <pthread.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "psrs.hpp"

#define M (1024 * 1024)

using namespace std;

static void init(vector<int>& data, mt19937& generator) {
    uniform_int_distribution<int> distribution(INT32_MIN, INT32_MAX);
    for (int& i : data) {
        i = distribution(generator);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: ./psrs <size_of_array> <number_of_threads>" << endl;
        return 1;
    }
    size_t size;
    size_t num_threads;
    try {
        size = stoull(argv[1]);
        num_threads = stoull(argv[2]);
    } catch (invalid_argument const& e) {
        cerr << "Invalid Argument. Arguments should be positive integers." << endl;
        return 1;
    } catch (out_of_range const& e) {
        cerr << "Out of Range. Arguments is too large." << endl;
        return 1;
    }
    cout << "Size of array: " << size << "M, Number of threads: " << num_threads << endl;
    size *= M;
    random_device rd;
    mt19937 generator(rd());
    vector<int> data(size);
    utils::Timer main_timer;
    init(data, generator);
    utils::print_vector(data);
    cout << "Data initialized." << endl;
    cout << "Sequential sorting started." << endl;
    vector<int> clone = data;
    main_timer.start();
    sort(clone.begin(), clone.end());
    main_timer.stop();
    cout << "Sequential sorting finished in " << main_timer.duration().count() << " milliseconds." << endl;
    cout << "Parallel sorting started." << endl;
    main_timer.start();
    vector<int> result = psrs::psrs(data, num_threads);
    main_timer.stop();
    cout << "Parallel sort finished in " << main_timer.duration().count() << " milliseconds." << endl;
    cout << "Checking result..." << endl;
    cout << (clone == result ? "Correct" : "Incorrect") << endl;
    return 0;
}
