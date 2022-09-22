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
#include <numeric>
#include <random>

#include "psrs.hpp"

#define K 1000
#define M (K * K)

using namespace std;

static void init(vector<int>& data, mt19937& generator) {
    auto distribution = uniform_int_distribution<int>(INT32_MIN, INT32_MAX);
    for (auto& i : data) {
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
    auto rd = random_device();
    auto generator = mt19937(rd());
    auto data = vector<int>(size);
    auto timer = utils::Timer<std::chrono::microseconds>();
    init(data, generator);
    cout << "Data initialized." << endl;
    cout << "Sequential sorting started." << endl;
    auto clone = vector<int>(data);
    timer.start();
    sort(clone.begin(), clone.end());
    timer.stop();
    cout << "Sequential sorting finished in " << timer.duration().count() << " microseconds." << endl;
    cout << "Parallel sorting started." << endl;
    auto elapsed_time = std::vector<int64_t>();
    elapsed_time.reserve(5);
    auto opt = optional<reference_wrapper<vector<int64_t>>>(elapsed_time);
    auto result = psrs::psrs(data, num_threads, opt);
    cout << "Parallel sorting finished in " << std::accumulate(elapsed_time.begin(), elapsed_time.end(), 0L)
         << " microseconds." << endl;
    cout << "Elapsed time in each phase: ";
    utils::print_vector(elapsed_time);
    cout << "Checking result..." << endl;
    cout << (clone == result ? "Correct" : "Incorrect") << endl;
    return 0;
}
