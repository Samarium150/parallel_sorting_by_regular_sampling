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
#include <random>

#include "psrs.hpp"

constexpr auto K = 1000;
constexpr auto M = K * K;

using namespace std;

static void init(vector<int>& data, mt19937& generator) {
    auto distribution = uniform_int_distribution<int>(INT32_MIN, INT32_MAX);
    for (auto& i : data) {
        i = distribution(generator);
    }
}

int main() {
    auto data = vector<int>(32 * M);
    auto rd = random_device();
    auto generator = mt19937(rd());
    init(data, generator);
    auto clone = vector(data);
    sort(clone.begin(), clone.end());
    auto result = parallel_sort(data, thread::hardware_concurrency());
    cout << "results of psrs is " << (result == clone ? "Correct" : "Wrong") << endl;
    return 0;
}
