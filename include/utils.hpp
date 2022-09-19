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

#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace utils {

    // https://stackoverflow.com/a/10758845
    template <typename T>
    [[maybe_unused]] void print_vector(std::vector<T>& vec) {
        std::ranges::copy(vec, std::ostream_iterator<T>(std::cout, " "));
        std::cout << std::endl;
    }

    // https://stackoverflow.com/a/21995693
    template <class DurationT = std::chrono::milliseconds, class ClockT = std::chrono::steady_clock>
    class Timer {
        using time_point = typename ClockT::time_point;

    protected:
        time_point _start = ClockT::now(), _end = {};

    public:
        void start() {
            _end = time_point{};
            _start = ClockT::now();
        }

        void stop() { _end = ClockT::now(); }

        template <class T = DurationT>
        auto duration() const {
            assert(_end != time_point{} && "stop the timer first");
            return std::chrono::duration_cast<T>(_end - _start);
        }
    };

    // https://stackoverflow.com/a/26221725
    template <typename... Args>
    std::string format(const std::string& format, Args... args) {
        int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'
        if (size_s <= 0) {
            throw std::runtime_error("Error during formatting.");
        }
        auto size = static_cast<size_t>(size_s);
        std::unique_ptr<char[]> buf(new char[size]);
        std::snprintf(buf.get(), size, format.c_str(), args...);
        return std::string{buf.get(), buf.get() + size - 1};  // We don't want the '\0' inside
    }
}  // namespace utils
