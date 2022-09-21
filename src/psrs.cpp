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
#include <cmath>
#include <iostream>

#include "psrs.hpp"

namespace psrs {

    static std::vector<int> merge_sorted_vectors(const std::vector<std::vector<int>>& vectors) {
        size_t size = vectors.size();
        size_t total_size = 0;
        for (size_t i = 0; i < size; i++) {
            total_size += vectors[i].size();
        }
        auto result = std::vector<int>(total_size);
        auto indices = std::vector<size_t>(size);
        size_t pos = 0;
        while (pos < total_size) {
            int min = INT32_MAX;
            size_t min_index = -1;
            for (size_t i = 0; i < size; i++) {
                if (indices[i] < vectors[i].size() && vectors[i][indices[i]] < min) {
                    min = vectors[i][indices[i]];
                    min_index = i;
                }
            }
            indices[min_index]++;
            result[pos++] = min;
        }
        return result;
    }

    static std::vector<Payload> init(const std::vector<int>& data, size_t num_threads, Globals& globals) {
        auto payloads = std::vector<Payload>();
        payloads.reserve(num_threads);
        size_t allocated = 0;
        for (size_t i = 0; i < num_threads; ++i) {
            size_t data_size = (i == num_threads - 1) ? data.size() - allocated : data.size() / num_threads;
            size_t stride_size = data.size() / (num_threads * num_threads);
            payloads.emplace_back(i, globals, data, allocated, allocated + data_size, stride_size);
            allocated += data_size;
        }
        return payloads;
    }

    static std::vector<int> phase_1(std::vector<int>& data, size_t stride_size) {
        std::sort(data.begin(), data.end());
        std::vector<int> samples;
        samples.reserve(data.size());
        for (size_t i = 1; i < data.size(); i += stride_size) {
            samples.emplace_back(data[i]);
        }
        return samples;
    }

    static void phase_2(const std::vector<std::vector<int>>& all_samples, std::vector<int>& pivots) {
        size_t num_threads = all_samples.size();
        auto sample_space = merge_sorted_vectors(all_samples);
        auto stride = (size_t)floor((double)num_threads / 2.0);
        for (size_t i = 1; i < num_threads; ++i) {
            pivots[i - 1] = sample_space[i * num_threads + stride];
        }
    }

    static void phase_3(size_t index,
                        const std::vector<int>& data,
                        const std::vector<int>& pivots,
                        partitions_t& all_partitions) {
        auto partitions = std::vector<std::vector<int>>(pivots.size() + 1);
        int prev = INT32_MIN;
        for (size_t i = 0; i < pivots.size(); ++i) {
            const auto& pivot = pivots[i];
            auto& partition = partitions[i];
            for (const int& value : data) {
                if (value > prev && value <= pivot) {
                    partition.emplace_back(value);
                } else if (value > pivot && i == pivots.size() - 1) {
                    partitions[i + 1].emplace_back(value);
                }
            }
            prev = pivot;
        }
        for (size_t i = 0; i < partitions.size(); ++i) {
            all_partitions[i][index] = partitions[i];
        }
    }

    static std::vector<int> phase_4(size_t index, partitions_t& all_partitions) {
        return merge_sorted_vectors(all_partitions[index]);
    }

    static void* psrs(void* arg) {
        auto payload = (Payload*)arg;
        auto& timer = payload->timer;
        auto& globals = payload->globals;
        pthread_barrier_wait(&globals.pthread_utils.p0_barrier);
        // Phase 1
        timer.start();
        globals.all_samples[payload->index] = phase_1(payload->data, payload->stride_size);
        timer.stop();
        pthread_mutex_lock(&globals.pthread_utils.mutex);
        globals.log_file << utils::format("1.%lld: %ld\n", payload->index, timer.duration().count());
#ifdef DEBUG
        std::cout << "Thread " << payload->index << " finished Phase 1" << std::endl;
#endif
        pthread_mutex_unlock(&globals.pthread_utils.mutex);
        pthread_barrier_wait(&globals.pthread_utils.p1_barrier);
        // Phase 2
        auto& pivots = globals.pivots;
        if (payload->index == 0) {
            timer.start();
            phase_2(globals.all_samples, pivots);
            timer.stop();
            pthread_mutex_lock(&globals.pthread_utils.mutex);
            globals.log_file << utils::format("2: %ld\n", timer.duration().count());
#ifdef DEBUG
            std::cout << "Thread 0 finished Phase 2" << std::endl;
#endif
            pthread_mutex_unlock(&globals.pthread_utils.mutex);
        }
        pthread_barrier_wait(&globals.pthread_utils.p2_barrier);
        // Phase 3
        timer.start();
        phase_3(payload->index, payload->data, pivots, globals.all_partitions);
        timer.stop();
        pthread_mutex_lock(&globals.pthread_utils.mutex);
        globals.log_file << utils::format("3.%lld: %ld\n", payload->index, timer.duration().count());
#ifdef DEBUG
        std::cout << "Thread " << payload->index << " finished Phase 3" << std::endl;
#endif
        pthread_mutex_unlock(&globals.pthread_utils.mutex);
        pthread_barrier_wait(&globals.pthread_utils.p3_barrier);
        // Phase 4
        timer.start();
        payload->result = phase_4(payload->index, globals.all_partitions);
        timer.stop();
        pthread_mutex_lock(&globals.pthread_utils.mutex);
        globals.log_file << utils::format("4.%lld: %ld\n", payload->index, timer.duration().count());
#ifdef DEBUG
        std::cout << "Thread " << payload->index << " finished Phase 4" << std::endl;
#endif
        pthread_mutex_unlock(&globals.pthread_utils.mutex);
        pthread_barrier_wait(&globals.pthread_utils.p4_barrier);
        return nullptr;
    }

    std::vector<int> psrs(const std::vector<int>& data, size_t num_threads) {
        auto timer = utils::Timer<std::chrono::microseconds>();
        auto log_file = std::ofstream(utils::format("psrs %lld %lld.txt", data.size(), num_threads), std::ios::out);
        auto pthread_utils = PthreadUtils(num_threads);
        auto threads = std::vector<pthread_t>(num_threads);
        auto globals = Globals::get_instance(num_threads, log_file, pthread_utils);
        auto payloads = init(data, num_threads, globals);
#ifdef DEBUG
        std::cout << "Initialization finished, starting threads..." << std::endl;
#endif
        threads[0] = pthread_self();
        for (size_t i = 1; i < num_threads; ++i) {
            if (pthread_create(&threads[i], &pthread_utils.attr, psrs, (void*)&payloads[i]) != 0) {
                std::cerr << "Failed to create thread " << i << "." << std::endl;
                log_file.close();
                exit(1);
            }
        }
        (void)psrs(&payloads[0]);
        log_file.close();
        std::vector<int> result;
        result.reserve(data.size());
        result.insert(result.end(), payloads[0].result.begin(), payloads[0].result.end());
        void* status;
        for (size_t i = 1; i < num_threads; ++i) {
            result.insert(result.end(), payloads[i].result.begin(), payloads[i].result.end());
            if (pthread_join(threads[i], &status) != 0) {
                std::cerr << "Failed to join thread " << i << "." << std::endl;
                exit(1);
            }
#ifdef DEBUG
            std::cout << "Thread " << i << " exited with " << status << std::endl;
#endif
        }
#ifdef DEBUG
        std::cout << "Thread 0 exited" << std::endl;
#endif
        return result;
    }
}  // namespace psrs
