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

using namespace std;

static void init(vector<int>& data, mt19937& generator) {
    auto distribution = uniform_int_distribution<int>(INT32_MIN, INT32_MAX);
    for (auto& i : data) {
        i = distribution(generator);
    }
}

static vector<int> run_sequential(const vector<int>& data, vector<int64_t>& time_records) {
    auto timer = utils::Timer<std::chrono::microseconds>();
    auto clone = vector<int>(data);
    for (size_t i = 0; i < 10; ++i) {
        timer.start();
        sort(clone.begin(), clone.end());
        timer.stop();
        auto t = timer.duration().count();
        time_records[i] = t;
        if (i != 9) {
            clone = data;
        }
    }
    return clone;
}

static vector<int> run_psrs(const vector<int>& data, size_t num_threads, vector<vector<int64_t>>& time_records) {
    vector<int> result;
    for (size_t i = 0; i < 10; ++i) {
        auto elapsed_time = std::vector<int64_t>();
        elapsed_time.reserve(6);
        auto opt = optional<reference_wrapper<vector<int64_t>>>(elapsed_time);
        result = psrs::psrs(data, num_threads, opt);
        time_records[i] = elapsed_time;
    }
    return result;
}

static void auto_tests() {
    for (size_t n = 0; n <= 5; ++n) {
        size_t size = (n == 0) ? 32 : n * 64;
        auto seq_log_file = ofstream(utils::format("sequential %lld.txt", size), ios::out);
        auto rd = random_device();
        auto generator = mt19937(rd());
        auto data = vector<int>(size * M);
        auto timer = utils::Timer<std::chrono::microseconds>();
        init(data, generator);
        auto sequential_time_records = vector<int64_t>(10);
        auto clone = run_sequential(data, sequential_time_records);
        auto sequential_average_time =
            std::accumulate(sequential_time_records.begin() + 5, sequential_time_records.end(), 0L) / 5;
        cout << "Sequential sorting finished in average of " << sequential_average_time << " microseconds for size "
             << size << "M" << endl;
        seq_log_file << sequential_average_time << endl;
        seq_log_file.close();
        for (size_t i = 1; i <= 10; ++i) {
            size_t num_threads = i * 2;
            auto parallel_log_file = ofstream(utils::format("parallel %lld %lld.txt", size, num_threads), ios::out);
            auto parallel_time_record = vector<vector<int64_t>>(10);
            vector<int> result = run_psrs(data, num_threads, parallel_time_record);
            if (result != clone) {
                cout << "psrs is incorrect in size " << size << "M with " << num_threads << "threads" << endl;
                exit(1);
            }
            auto parallel_phases_average_time = vector<int64_t>(6);
            for (size_t j = 0; j < 6; ++j) {
                auto sum = 0L;
                for (size_t k = 0; k < 10; ++k) {
                    sum += parallel_time_record[k][j];
                }
                parallel_phases_average_time[j] = sum / 10;
            }
            auto parallel_average_time =
                std::accumulate(parallel_phases_average_time.begin(), parallel_phases_average_time.end(), 0L);
            cout << "Parallel sorting finished in average of " << parallel_average_time << " microseconds for size "
                 << size << "M with " << num_threads << "threads" << std::endl;
            for (size_t j = 0; j < 6; ++j) {
                parallel_log_file << "p." << j << ": " << parallel_phases_average_time[j] << endl;
            }
            parallel_log_file << parallel_average_time << endl;
            parallel_log_file.close();
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc == 2 && string(argv[1]) == "auto") {
        cout << "Running auto tests..." << endl;
        auto_tests();
        return 0;
    }
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
    auto log_file = ofstream(utils::format("psrs %lld %lld.txt", size, num_threads), ios::out);
    size *= M;
    auto rd = random_device();
    auto generator = mt19937(rd());
    auto data = vector<int>(size);
    auto timer = utils::Timer<std::chrono::microseconds>();
    init(data, generator);
    cout << "Data initialized." << endl;
    cout << "Sequential sorting started. (10 times)" << endl;
    auto sequential_time_record = vector<int64_t>(10);
    auto clone = run_sequential(data, sequential_time_record);
    auto sequential_average_time =
        std::accumulate(sequential_time_record.begin() + 5, sequential_time_record.end(), 0L) / 5;
    cout << "Sequential sorting finished in average of " << sequential_average_time << " microseconds." << endl;
    log_file << "s: " << sequential_average_time << endl;
    cout << "Parallel sorting started. (10 times)" << endl;
    auto parallel_time_record = vector<vector<int64_t>>(10);
    vector<int> result = run_psrs(data, num_threads, parallel_time_record);
    auto parallel_phases_average_time = vector<int64_t>(6);
    for (size_t i = 0; i < 6; ++i) {
        auto sum = 0L;
        for (size_t j = 0; j < 10; ++j) {
            sum += parallel_time_record[j][i];
        }
        parallel_phases_average_time[i] = sum / 10;
    }
    auto parallel_average_time =
        std::accumulate(parallel_phases_average_time.begin(), parallel_phases_average_time.end(), 0L);
    cout << "Parallel sorting finished in average of " << parallel_average_time << " microseconds." << endl;
    cout << "Average elapsed time in each phase: ";
    utils::print_vector(parallel_phases_average_time);
    for (size_t i = 0; i < 6; ++i) {
        log_file << "p." << i << ": " << parallel_phases_average_time[i] << endl;
    }
    log_file << "p: " << parallel_average_time << endl;
    cout << "Checking result..." << endl;
    cout << (clone == result ? "Correct" : "Incorrect") << endl;
    log_file.close();
    return 0;
}
