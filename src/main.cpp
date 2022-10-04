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
#include <fstream>
#include <numeric>
#include <random>

#include "psrs.hpp"

#define K 1000
#define M (K * K)
#define TIMES 10

using namespace std;

static void init(vector<int>& data, mt19937& generator) {
    auto distribution = uniform_int_distribution<int>(INT32_MIN, INT32_MAX);
    for (auto& i : data) {
        i = distribution(generator);
    }
}

static void auto_test(size_t size, optional<size_t> num_threads = {}) {
    auto rd = random_device();
    auto generator = mt19937(rd());
    auto data = vector<int>(size * M);
    auto timer = utils::Timer<std::chrono::microseconds>();
    auto sequential_time_records = vector<int64_t>(10);
    auto parallel_time_records = vector<vector<vector<int64_t>>>(10);
    auto total_threads = (size_t)sysconf(_SC_NPROCESSORS_ONLN);
    for (size_t i = 0; i < TIMES; ++i) {
        init(data, generator);
        auto clone = vector(data);
        cout << "Trial No." << i + 1 << ": " << endl;
        cout << "run sequential for " << size << "M" << endl;
        timer.start();
        sort(clone.begin(), clone.end());
        timer.stop();
        auto time = timer.duration().count();
        sequential_time_records[i] = time;
        if (num_threads.has_value()) {
            size_t threads = num_threads.value();
            auto threads_time_record = vector<vector<int64_t>>(1);
            cout << "run psrs with " << threads << " threads" << endl;
            auto elapsed_time = vector<int64_t>();
            elapsed_time.reserve(6);
            auto opt = optional<reference_wrapper<vector<int64_t>>>(elapsed_time);
            auto result = psrs::psrs(vector(data), threads, opt);
            if (result != clone) {
                cout << "psrs is incorrect in size " << size << "M with " << threads << "threads" << endl;
                exit(1);
            }
            threads_time_record[0] = elapsed_time;
            parallel_time_records[i] = threads_time_record;
        } else {
            auto threads_time_record = vector<vector<int64_t>>(total_threads / 2);
            for (size_t t = 1; t <= total_threads / 2; ++t) {
                size_t threads = t * 2;
                cout << "run psrs with " << threads << " threads" << endl;
                auto elapsed_time = vector<int64_t>();
                elapsed_time.reserve(6);
                auto opt = optional<reference_wrapper<vector<int64_t>>>(elapsed_time);
                auto result = psrs::psrs(vector(data), threads, opt);
                if (result != clone) {
                    cout << "psrs is incorrect in size " << size << "M with " << threads << "threads" << endl;
                    exit(1);
                }
                threads_time_record[t - 1] = elapsed_time;
            }
            parallel_time_records[i] = threads_time_record;
        }
    }
    size_t sequential_average_time =
        accumulate(sequential_time_records.begin() + 5, sequential_time_records.end(), 0L) / 5;
    cout << "Sequential sorting finished in average of " << sequential_average_time << " microseconds for size " << size
         << "M" << endl;
    auto seq_log_file = ofstream(utils::format("sequential %lld.txt", size), ios::out);
    seq_log_file << sequential_average_time << endl;
    seq_log_file.close();
    if (num_threads.has_value()) {
        size_t threads = num_threads.value();
        auto parallel_phases_average_time = vector<int64_t>(6);
        for (size_t i = TIMES - 5; i < TIMES; ++i) {
            for (size_t p = 0; p < 6; ++p) {
                parallel_phases_average_time[p] += parallel_time_records[i][0][p];
            }
        }
        for (size_t p = 0; p < 6; ++p) {
            parallel_phases_average_time[p] /= 5;
        }
        int64_t average_time = accumulate(parallel_phases_average_time.begin(), parallel_phases_average_time.end(), 0L);
        auto parallel_log_file = ofstream(utils::format("parallel %lld %lld.txt", size, threads), ios::out);
        for (size_t p = 0; p < 6; ++p) {
            parallel_log_file << "p." << p << ": " << parallel_phases_average_time[p] << endl;
        }
        cout << "Parallel sorting finished in average of " << average_time << " microseconds for size " << size
             << "M with " << threads << "threads" << std::endl;
        parallel_log_file << average_time << endl;
        parallel_log_file.close();
    } else {
        auto parallel_phases_average_time = vector<vector<int64_t>>(total_threads / 2);
        for (size_t t = 0; t < total_threads / 2; ++t) {
            auto time = vector<int64_t>(6);
            for (size_t i = TIMES - 5; i < TIMES; ++i) {
                for (size_t p = 0; p < 6; ++p) {
                    time[p] += parallel_time_records[i][t][p];
                }
            }
            for (size_t p = 0; p < 6; ++p) {
                time[p] /= 5;
            }
            parallel_phases_average_time[t] = time;
        }
        for (size_t t = 0; t < total_threads / 2; ++t) {
            size_t threads = (t + 1) * 2;
            int64_t average_time =
                accumulate(parallel_phases_average_time[t].begin(), parallel_phases_average_time[t].end(), 0L);
            auto parallel_log_file = ofstream(utils::format("parallel %lld %lld.txt", size, threads), ios::out);
            for (size_t p = 0; p < 6; ++p) {
                parallel_log_file << "p." << p << ": " << parallel_phases_average_time[t][p] << endl;
            }
            cout << "Parallel sorting finished in average of " << average_time << " microseconds for size " << size
                 << "M with " << threads << " threads" << std::endl;
            parallel_log_file << average_time << endl;
            parallel_log_file.close();
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2 && argc != 3) {
        cout << "Usage: ./psrs <size_of_array> [number_of_threads]" << endl;
        return 1;
    }
    if (argc == 2) {
        size_t size;
        try {
            size = stoull(argv[1]);
        } catch (const std::invalid_argument& e) {
            cout << "Invalid argument: " << argv[1] << endl;
            return 1;
        }
        auto_test(size);
        return 0;
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
    auto_test(size, num_threads);
    return 0;
}
