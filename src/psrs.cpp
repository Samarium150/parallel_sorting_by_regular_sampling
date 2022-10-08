/*
 * Copyright (C) 2022 Samarium
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
#include <barrier>
#include <cmath>
#include <mutex>

#include "psrs.hpp"

namespace psrs {

    using partitions_t = typename std::vector<std::vector<std::vector<int>>>;

    class alignas(64) ThreadUtils {
    public:
        std::barrier<> b0;
        std::barrier<> b1;
        std::barrier<> b2;
        std::barrier<> b3;
        std::barrier<> b4;
        std::mutex mutex{};
        explicit ThreadUtils(ptrdiff_t num_threads)
            : b0(num_threads), b1(num_threads), b2(num_threads), b3(num_threads), b4(num_threads) {}
    };

    class alignas(64) Globals {
    public:
        ThreadUtils thread_utils;
        std::vector<int> pivots;
        std::vector<std::vector<int>> all_samples;
        partitions_t all_partitions;
        explicit Globals(ptrdiff_t num_threads) : thread_utils(num_threads) {
            pivots = std::vector<int>(num_threads - 1);
            all_samples = std::vector<std::vector<int>>(num_threads);
            all_partitions = partitions_t(num_threads);
            for (auto& partitions : all_partitions) {
                partitions.reserve(num_threads);
                for (ptrdiff_t _ = 0; _ < num_threads; ++_) {
                    partitions.emplace_back(std::vector<int>());
                }
            }
        }
    };

    class alignas(64) Payload {
    public:
        size_t index;
        Globals& globals;
        std::vector<int> data;
        size_t stride_size;
        std::vector<int> result{};
        Payload(size_t index,
                Globals& globals,
                const std::vector<int>& data,
                size_t begin,
                size_t end,
                size_t stride_size)
            : index(index), globals(globals), stride_size(stride_size) {
            auto first = data.begin() + (long)begin;
            auto last = data.begin() + (long)end;
            this->data = std::vector<int>(first, last);
        }
    };

    static std::vector<int> merge_sorted_vectors(const std::vector<std::vector<int>>& vectors) {
        size_t size = vectors.size();
        size_t total_size = 0;
        for (size_t i = 0; i < size; ++i) {
            total_size += vectors[i].size();
        }
        auto result = std::vector<int>(total_size);
        auto indices = std::vector<size_t>(size);
        size_t pos = 0;
        while (pos < total_size) {
            int min = INT32_MAX;
            size_t min_index = -1;
            for (size_t i = 0; i < size; ++i) {
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

    static void phase_3(size_t id,
                        const std::vector<int>& data,
                        const std::vector<int>& pivots,
                        partitions_t& all_partitions) {
        int index = 0;
        for (size_t i = 0; i < pivots.size(); ++i) {
            auto it = std::lower_bound(data.begin() + index, data.end(), pivots[i]);
            all_partitions[i][id] = std::vector(data.begin() + index, it);
            index = (int)(it - data.begin());
        }
        all_partitions[pivots.size()][id] = std::vector(data.begin() + index, data.end());
    }

    static std::vector<int> phase_4(size_t id, partitions_t& all_partitions) {
        return merge_sorted_vectors(all_partitions[id]);
    }

    static void psrs(Payload& payload) {
        auto& globals = payload.globals;
        globals.thread_utils.b0.arrive_and_wait();
        // Phase 1
        globals.all_samples[payload.index] = phase_1(payload.data, payload.stride_size);
#ifdef _DEBUG
        globals.thread_utils.mutex.lock();
        std::cout << "Thread " << payload.index << " finished Phase 1" << std::endl;
        globals.thread_utils.mutex.unlock();
#endif
        globals.thread_utils.b1.arrive_and_wait();
        // Phase 2
        auto& pivots = globals.pivots;
        if (payload.index == 0) {
            phase_2(globals.all_samples, pivots);
#ifdef _DEBUG
            std::cout << "Thread 0 finished Phase 2" << std::endl;
#endif
        }
        globals.thread_utils.b2.arrive_and_wait();
        // Phase 3
        phase_3(payload.index, payload.data, pivots, globals.all_partitions);
#ifdef _DEBUG
        globals.thread_utils.mutex.lock();
        std::cout << "Thread " << payload.index << " finished Phase 3" << std::endl;
        globals.thread_utils.mutex.unlock();
#endif
        globals.thread_utils.b3.arrive_and_wait();
        // Phase 4
        payload.result = phase_4(payload.index, globals.all_partitions);
#ifdef _DEBUG
        globals.thread_utils.mutex.lock();
        std::cout << "Thread " << payload.index << " finished Phase 4" << std::endl;
        globals.thread_utils.mutex.unlock();
#endif
        globals.thread_utils.b4.arrive_and_wait();
    }
}  // namespace psrs

std::vector<int> parallel_sort(const std::vector<int>& data, size_t num_threads) {
    auto globals = psrs::Globals((ptrdiff_t)num_threads);
    auto threads = std::vector<std::thread>();
    auto payloads = psrs::init(data, num_threads, globals);
    for (size_t i = 1; i < num_threads; ++i) {
        threads.emplace_back(psrs::psrs, std::ref(payloads[i]));
    }
    psrs::psrs(payloads[0]);
    std::vector<int> result;
    result.reserve(data.size());
    result.insert(result.end(), payloads[0].result.begin(), payloads[0].result.end());
    for (size_t i = 1; i < num_threads; ++i) {
        threads[i - 1].join();
        result.insert(result.end(), payloads[i].result.begin(), payloads[i].result.end());
    }
    return result;
}
