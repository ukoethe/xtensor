/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xtiny_experiment.hpp"
#include "xtensor/xtiny.hpp"
#include "xtensor/xarray.hpp"
#include <benchmark/benchmark.h>

static void BM_TinyExperimentCreation(benchmark::State& state) {
    using array = xt::xtiny_array_experiment<int, 2>;

    array a;
    a[0] = 1;
    a[1] = 2;

    for (auto _ : state)
        array b = a + a*a;
}
// Register the function as a benchmark
BENCHMARK(BM_TinyExperimentCreation);

static void BM_TinyCreation(benchmark::State& state) {
    using array = xt::tiny_array<int, 2>;

    array a;
    a[0] = 1;
    a[1] = 2;

    for (auto _ : state)
        array b = a + a*a;
}
// Register the function as a benchmark
BENCHMARK(BM_TinyCreation);

namespace xt
{
    TEST(xtiny, xarray)
    {
        using array = xarray<int>;

        array a = {1, 2};

        EXPECT_EQ(a.shape(), (std::vector<size_t>{2}));
        EXPECT_EQ(a[0], 1);
        EXPECT_EQ(a[1], 2);

        array b = a + a*a;
        EXPECT_EQ(b[0], 2);
        EXPECT_EQ(b[1], 6);

        auto c = 2*b;
        EXPECT_EQ(c(), 4);
        EXPECT_EQ(c(0), 4);
        EXPECT_EQ(c(1), 12);
    }

    TEST(xtiny, xtiny_array)
    {
        using array = xtiny_array_experiment<int, 2>;

        array a;

        EXPECT_EQ(a.shape(), (std::array<size_t, 1>{2}));
        a[0] = 1;
        a[1] = 2;
        EXPECT_EQ(a[0], 1);
        EXPECT_EQ(a[1], 2);

        array b = a + a*a;
        EXPECT_EQ(b[0], 2);
        EXPECT_EQ(b[1], 6);

        auto c = 2*b;
        EXPECT_EQ(c(), 4);
        EXPECT_EQ(c(0), 4);
        EXPECT_EQ(c(1), 12);
    }

    TEST(xtiny, tiny_array)
    {
        using array = tiny_array<int, 2>;

        array a;

        EXPECT_EQ(a.shape(), (std::array<index_t, 1>{2}));
        a[0] = 1;
        a[1] = 2;
        EXPECT_EQ(a[0], 1);
        EXPECT_EQ(a[1], 2);

        array b = a + a*a;
        EXPECT_EQ(b[0], 2);
        EXPECT_EQ(b[1], 6);

        auto c = 2*b;
        EXPECT_EQ(c(0), 4);
        EXPECT_EQ(c(1), 12);
    }
} // namespace xt
