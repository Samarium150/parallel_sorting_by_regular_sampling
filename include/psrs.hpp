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
#include <fstream>
#include <utility>
#include <vector>

#include "utils.hpp"

namespace psrs {

    class alignas(64) PthreadUtils {
    public:
        pthread_attr_t attr{};
        pthread_mutex_t mutex{};
        pthread_barrier_t p1_barrier{};
        pthread_barrier_t p2_barrier{};
        pthread_barrier_t p3_barrier{};
        pthread_barrier_t p4_barrier{};
        explicit PthreadUtils(size_t num_threads) {
            pthread_attr_init(&attr);
            pthread_mutex_init(&mutex, nullptr);
            pthread_barrier_init(&p1_barrier, nullptr, num_threads + 1);
            pthread_barrier_init(&p2_barrier, nullptr, num_threads + 1);
            pthread_barrier_init(&p3_barrier, nullptr, num_threads + 1);
            pthread_barrier_init(&p4_barrier, nullptr, num_threads + 1);
        }
        ~PthreadUtils() {
            pthread_attr_destroy(&attr);
            pthread_mutex_destroy(&mutex);
            pthread_barrier_destroy(&p1_barrier);
            pthread_barrier_destroy(&p2_barrier);
            pthread_barrier_destroy(&p3_barrier);
            pthread_barrier_destroy(&p4_barrier);
        }
    };

    class alignas(64) P1Payload {
    public:
        std::vector<int> data;
        std::vector<int> samples;
        size_t stride_size;
        P1Payload(const std::vector<int>& data, size_t begin, size_t end, size_t stride_size)
            : stride_size(stride_size) {
            auto first = data.begin() + (long)begin;
            auto last = data.begin() + (long)end;
            this->data = std::vector<int>(first, last);
        }
    };

    class alignas(64) P3Payload {
    public:
        std::vector<int>* __restrict data{};
        std::vector<int>* __restrict pivots;
        std::vector<std::vector<int>*> partition_ptrs;
        P3Payload(std::vector<int>* __restrict pivots, std::vector<std::vector<int>*> partition_ptrs)
            : pivots(pivots), partition_ptrs(std::move(partition_ptrs)) {}
    };

    class alignas(64) P4Payload {
    public:
        std::vector<std::vector<std::vector<int>>>* all_partitions;
        std::vector<int> result;
        explicit P4Payload(std::vector<std::vector<std::vector<int>>>* all_partitions)
            : all_partitions(all_partitions) {}
    };

    class alignas(64) Payload {
    public:
        size_t index;
        utils::Timer<std::chrono::microseconds> timer{};
        std::ofstream* log_file;
        PthreadUtils* pthread_utils;
        P1Payload p1_payload;
        P3Payload p3_payload;
        P4Payload p4_payload;
        Payload(size_t index, std::ofstream* log_file, PthreadUtils* pu, P1Payload p1p, P3Payload p3p, P4Payload p4p)
            : index(index), log_file(log_file), pthread_utils(pu), p1_payload(std::move(p1p)),
              p3_payload(std::move(p3p)), p4_payload(std::move(p4p)) {}
    };

    std::vector<int> psrs(const std::vector<int>&, size_t);
}  // namespace psrs
