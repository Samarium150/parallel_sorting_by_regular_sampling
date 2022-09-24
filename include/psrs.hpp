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
#pragma once

#include <pthread.h>
#include <optional>

#include "utils.hpp"

namespace psrs {

    using partitions_t = typename std::vector<std::vector<std::vector<int>>>;
    using optional_elapsed_time_records = typename std::optional<std::reference_wrapper<std::vector<int64_t>>>;

    class alignas(64) PthreadUtils {
    public:
        pthread_attr_t attr{};
        pthread_barrier_t p0_barrier{};
        pthread_barrier_t p1_barrier{};
        pthread_barrier_t p2_barrier{};
        pthread_barrier_t p3_barrier{};
        pthread_barrier_t p4_barrier{};
        explicit PthreadUtils(size_t num_threads) {
            pthread_attr_init(&attr);
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
            pthread_barrier_init(&p0_barrier, nullptr, num_threads);
            pthread_barrier_init(&p1_barrier, nullptr, num_threads);
            pthread_barrier_init(&p2_barrier, nullptr, num_threads);
            pthread_barrier_init(&p3_barrier, nullptr, num_threads);
            pthread_barrier_init(&p4_barrier, nullptr, num_threads);
        }
        ~PthreadUtils() {
            pthread_attr_destroy(&attr);
            pthread_barrier_destroy(&p0_barrier);
            pthread_barrier_destroy(&p1_barrier);
            pthread_barrier_destroy(&p2_barrier);
            pthread_barrier_destroy(&p3_barrier);
            pthread_barrier_destroy(&p4_barrier);
        }
    };

    class alignas(64) Globals {
    public:
        PthreadUtils& pthread_utils;
        std::vector<int> pivots;
        std::vector<std::vector<int>> all_samples;
        partitions_t all_partitions;
        Globals(size_t num_threads, PthreadUtils& pthread_utils) : pthread_utils(pthread_utils) {
            pivots = std::vector<int>(num_threads - 1);
            all_samples = std::vector<std::vector<int>>(num_threads);
            all_partitions = partitions_t(num_threads);
            for (auto& partitions : all_partitions) {
                partitions.reserve(num_threads);
                for (size_t _ = 0; _ < num_threads; ++_) {
                    partitions.emplace_back(std::vector<int>());
                }
            }
        }
    };

    class alignas(64) Payload {
    public:
        size_t index;
        utils::Timer<std::chrono::microseconds> timer{};
        Globals& globals;
        std::vector<int> data;
        size_t stride_size;
        std::vector<int> result{};
        std::vector<int64_t> elapsed_time{};
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
            this->elapsed_time.reserve(4);
        }
    };

    std::vector<int> psrs(const std::vector<int>&, size_t, optional_elapsed_time_records);
}  // namespace psrs
