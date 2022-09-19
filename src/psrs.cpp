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

    static std::vector<int> merge_sorted_vectors(std::vector<std::vector<int>> vectors) {
        size_t size = vectors.size();
        size_t total_size = 0;
        for (size_t i = 0; i < size; i++) {
            total_size += vectors[i].size();
        }
        std::vector<int> result(total_size);
        std::vector<size_t> indices(size);
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

    static void init(const std::vector<int>& data,
                     size_t num_threads,
                     std::ofstream& log_file,
                     PthreadUtils& pthread_utils,
                     std::vector<int>& pivots,
                     std::vector<std::vector<std::vector<int>>>& all_partitions,
                     std::vector<Payload>& payloads) {
        pthread_attr_setdetachstate(&pthread_utils.attr, PTHREAD_CREATE_JOINABLE);
        std::vector<std::vector<std::vector<int>*>> all_partitions_ptr(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            all_partitions[i].reserve(num_threads);
            for (size_t j = 0; j < num_threads; ++j) {
                all_partitions[i].emplace_back(std::vector<int>());
            }
        }
        for (size_t i = 0; i < num_threads; ++i) {
            for (size_t j = 0; j < num_threads; ++j) {
                all_partitions_ptr[i].emplace_back(&all_partitions[i][j]);
            }
        }
        size_t allocated = 0;
        for (size_t i = 0; i < num_threads; ++i) {
            size_t data_size = (i == num_threads - 1) ? data.size() - allocated : data.size() / num_threads;
            size_t stride_size = data.size() / (num_threads * num_threads);
            P1Payload p1_payload(data, allocated, allocated + data_size, stride_size);
            P3Payload p3_payload(&pivots, all_partitions_ptr[i]);
            P4Payload p4_payload(&all_partitions);
            payloads.emplace_back(i, &log_file, &pthread_utils, p1_payload, p3_payload, p4_payload);
            allocated += data_size;
        }
    }

    static void phase_1(P1Payload& payload) {
        std::vector<int>& data = payload.data;
        std::sort(data.begin(), data.end());
        for (size_t i = 1; i < data.size(); i += payload.stride_size) {
            payload.samples.emplace_back(data[i]);
        }
    }

    static void phase_2(const std::vector<std::vector<int>>& samples, std::vector<int>& pivots) {
        size_t num_threads = samples.size();
        std::vector<int> sample_space = merge_sorted_vectors(samples);
        auto stride = (size_t)floor((double)num_threads / 2.0);
        for (size_t i = 1; i < num_threads; ++i) {
            pivots[i - 1] = sample_space[i * num_threads + stride];
        }
    }

    static void phase_3(P3Payload& payload) {
        std::vector<int>& data = *payload.data;
        std::vector<int>& pivots = *payload.pivots;
        std::vector<std::vector<int>> partitions(pivots.size() + 1);
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
            payload.partition_ptrs[i]->assign(partitions[i].begin(), partitions[i].end());
        }
    }

    static void phase_4(size_t index, P4Payload& payload) {
        size_t num_threads = payload.all_partitions->size();
        std::vector<std::vector<int>> partitions(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            partitions[i] = (*payload.all_partitions)[i][index];
        }
        payload.result = merge_sorted_vectors(partitions);
    }

    static void* psrs(void* arg) {
        auto* payload = (Payload*)arg;
        auto* pu = payload->pthread_utils;
        auto& timer = payload->timer;
        std::ofstream& log_file = *payload->log_file;
        timer.start();
        phase_1(payload->p1_payload);
        timer.stop();
        pthread_mutex_lock(&pu->mutex);
        log_file << utils::format("1.%lld: %ld\n", payload->index, timer.duration().count());
        pthread_mutex_unlock(&pu->mutex);
        pthread_barrier_wait(&pu->p1_barrier);
        pthread_barrier_wait(&pu->p2_barrier);
        timer.start();
        payload->p3_payload.data = &payload->p1_payload.data;
        phase_3(payload->p3_payload);
        timer.stop();
        pthread_mutex_lock(&pu->mutex);
        log_file << utils::format("3.%lld: %ld\n", payload->index, timer.duration().count());
        pthread_mutex_unlock(&pu->mutex);
        pthread_barrier_wait(&pu->p3_barrier);
        timer.start();
        phase_4(payload->index, payload->p4_payload);
        timer.stop();
        pthread_barrier_wait(&pu->p4_barrier);
        pthread_mutex_lock(&pu->mutex);
        log_file << utils::format("4.%lld: %ld\n", payload->index, timer.duration().count());
        pthread_mutex_unlock(&pu->mutex);
        return nullptr;
    }

    std::vector<int> psrs(const std::vector<int>& data, size_t num_threads) {
        utils::Timer<std::chrono::microseconds> timer;
        std::ofstream log_file(utils::format("psrs %lld %lld.txt", data.size(), num_threads), std::ios::out);
        PthreadUtils pthread_utils(num_threads);
        std::vector<pthread_t> threads(num_threads);
        std::vector<std::vector<std::vector<int>>> all_partitions(num_threads);
        std::vector<int> pivots(num_threads - 1);
        std::vector<Payload> payloads;
        init(data, num_threads, log_file, pthread_utils, pivots, all_partitions, payloads);
        std::cout << "Initialization finished, starting threads" << std::endl;
        for (size_t i = 0; i < num_threads; ++i) {
            if (pthread_create(&threads[i], &pthread_utils.attr, psrs, (void*)&payloads[i]) != 0) {
                std::cerr << "Failed to create Thread " << i << "." << std::endl;
                log_file.close();
                exit(1);
            }
        }
        pthread_barrier_wait(&pthread_utils.p1_barrier);
        std::cout << "Phase 1: All threads finished." << std::endl;
        timer.start();
        std::vector<std::vector<int>> samples(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            samples[i] = payloads[i].p1_payload.samples;
        }
        phase_2(samples, pivots);
        timer.stop();
        log_file << utils::format("2: %ld\n", timer.duration().count());
        pthread_barrier_wait(&pthread_utils.p2_barrier);
        std::cout << "Phase 2 finished." << std::endl;
        pthread_barrier_wait(&pthread_utils.p3_barrier);
        std::cout << "Phase 3: All threads finished." << std::endl;
        pthread_barrier_wait(&pthread_utils.p4_barrier);
        std::cout << "Phase 4: All threads finished." << std::endl;
        log_file.close();
        pthread_utils.~PthreadUtils();
        std::vector<int> result;
        void* status;
        for (size_t i = 0; i < num_threads; ++i) {
            result.insert(result.end(), payloads[i].p4_payload.result.begin(), payloads[i].p4_payload.result.end());
            if (pthread_join(threads[i], &status) != 0) {
                std::cerr << "Failed to join Thread " << i << "." << std::endl;
                exit(1);
            }
            std::cout << "Thread " << i << " exiting with " << status << std::endl;
        }
        std::cout << "PSRS finished." << std::endl;
        return result;
    }
}  // namespace psrs
