/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include <initializer_list>
#include <type_traits>
#include <tuple>
#include <complex>
#include "xtensor/xutils.hpp"

namespace xt
{
    using std::size_t;

    struct for_each_fn
    {
        short a;
        int b;
        float c;
        double d;

        template <class T>
        void operator()(T t)
        {
            if (std::is_same<T, short>::value)
            {
                a = static_cast<short>(t);
            }
            else if (std::is_same<T, int>::value)
            {
                b = static_cast<int>(t);
            }
            else if (std::is_same<T, float>::value)
            {
                c = static_cast<float>(t);
            }
            else if (std::is_same<T, double>::value)
            {
                d = static_cast<double>(t);
            }
        }
    };

    TEST(utils, for_each)
    {
        for_each_fn fn;
        short a = 1;
        int b = 4;
        float c = float(1.2);
        double d = 2.3;
        auto t = std::make_tuple(a, b, c, d);
        for_each(fn, t);
        ASSERT_TRUE(a == fn.a && b == fn.b && c == fn.c && d == fn.d);
    }

    TEST(utils, accumulate)
    {
        auto func = [](int i, int j) { return i + j; };
        const std::tuple<int, int, int> t(3, 4, 1);
        EXPECT_EQ(8, accumulate(func, 0, t));
    }

    TEST(utils, or)
    {
        using true_t = std::true_type;
        using false_t = std::false_type;

        using t1 = or_<false_t, false_t, false_t>;
        using t2 = or_<false_t, true_t, false_t>;

        ASSERT_TRUE(!t1::value);
        ASSERT_TRUE(t2::value);
    }

    template <class... T>
    auto foo(const std::tuple<T...>& t)
    {
        auto func = [](int i) { return i; };
        return apply<int>(1, func, t);
    }

    TEST(utils, apply)
    {
        ASSERT_TRUE(foo(std::make_tuple(1, 2, 3)) == 2);
    }

    TEST(utils, initializer_dimension)
    {
        size_t d0 = initializer_dimension<double>::value;
        size_t d1 = initializer_dimension<std::initializer_list<double>>::value;
        size_t d2 = initializer_dimension<std::initializer_list<std::initializer_list<double>>>::value;
        EXPECT_EQ(size_t(0), d0);
        EXPECT_EQ(size_t(1), d1);
        EXPECT_EQ(size_t(2), d2);
    }

    TEST(utils, promote_shape)
    {
        bool expect_v = std::is_same<
            std::vector<size_t>,
            promote_shape_t<std::vector<size_t>, std::array<size_t, 3>, std::array<size_t, 2>>
        >::value;

        bool expect_a = std::is_same<
            std::array<size_t, 3>,
            promote_shape_t<std::array<size_t, 2>, std::array<size_t, 3>, std::array<size_t, 2>>
        >::value;

        ASSERT_TRUE(expect_v);
        ASSERT_TRUE(expect_a);
    }

    TEST(utils, shape)
    {
        auto s0 = shape<std::vector<size_t>>(3);
        auto s1 = shape<std::vector<size_t>>(std::initializer_list<size_t>{1, 2});
        auto s2 = shape<std::vector<size_t>>(std::initializer_list<std::initializer_list<size_t>>{{1, 2, 4}, {1, 3, 5}});

        std::vector<size_t> e0 = {};
        std::vector<size_t> e1 = {2};
        std::vector<size_t> e2 = {2, 3};

        ASSERT_TRUE(check_shape(3, s0.begin(), s0.end()));
        ASSERT_TRUE(check_shape(std::initializer_list<size_t>{1, 2}, s1.begin(), s1.end()));
        ASSERT_TRUE(check_shape(std::initializer_list<std::initializer_list<size_t>>{{1, 2, 4}, {1, 3, 5}}, s2.begin(), s2.end()));

        EXPECT_EQ(e0, s0);
        EXPECT_EQ(e1, s1);
        EXPECT_EQ(e2, s2);
    }

    TEST(utils, forward_offset)
    {
        // Test that lvalues can be modified
        std::complex<double> clv;
        forward_real(clv) = 3.0;
        EXPECT_EQ(std::real(clv), 3.0);

        forward_imag(clv) = 1.0;
        EXPECT_EQ(std::imag(clv), 1.0);

        double rlv = 2.0;
        forward_real(rlv) = 1.0;
        EXPECT_EQ(forward_imag(rlv), 0.0);
        EXPECT_EQ(forward_real(rlv), 1.0);
    }

    TEST(utils, conditional_cast)
    {
        EXPECT_TRUE((std::is_same<decltype(conditional_cast<false, double>(1)), int>::value));
        EXPECT_TRUE((std::is_same<decltype(conditional_cast<true, double>(1)), double>::value));
    }

    TEST(utils, promote_traits)
    {
        EXPECT_TRUE((std::is_same<promote_type_t<uint8_t>, int>::value));
        EXPECT_TRUE((std::is_same<promote_type_t<int>, int>::value));
        EXPECT_TRUE((std::is_same<promote_type_t<float>, float>::value));
        EXPECT_TRUE((std::is_same<promote_type_t<double>, double>::value));

        EXPECT_TRUE((std::is_same<real_promote_type_t<uint8_t>, double>::value));
        EXPECT_TRUE((std::is_same<real_promote_type_t<int>, double>::value));
        EXPECT_TRUE((std::is_same<real_promote_type_t<float>, float>::value));
        EXPECT_TRUE((std::is_same<real_promote_type_t<double>, double>::value));

        EXPECT_TRUE((std::is_same<bool_promote_type_t<bool>, uint8_t>::value));
        EXPECT_TRUE((std::is_same<bool_promote_type_t<int>, int>::value));
    }

    TEST(utils, norm_traits)
    {
        EXPECT_TRUE((std::is_same<norm_type_t<uint8_t>, uint8_t>::value));
        EXPECT_TRUE((std::is_same<norm_type_t<int>, int>::value));
        EXPECT_TRUE((std::is_same<norm_type_t<double>, double>::value));
        EXPECT_TRUE((std::is_same<norm_type_t<std::vector<uint8_t>>, double>::value));
        EXPECT_TRUE((std::is_same<norm_type_t<std::vector<int>>, double>::value));
        EXPECT_TRUE((std::is_same<norm_type_t<std::vector<double>>, double>::value));
        EXPECT_TRUE((std::is_same<norm_type_t<std::vector<long double>>, long double>::value));

        EXPECT_TRUE((std::is_same<squared_norm_type_t<uint8_t>, int>::value));
        EXPECT_TRUE((std::is_same<squared_norm_type_t<int>, int>::value));
        EXPECT_TRUE((std::is_same<squared_norm_type_t<double>, double>::value));
        EXPECT_TRUE((std::is_same<squared_norm_type_t<std::vector<uint8_t>>, uint64_t>::value));
        EXPECT_TRUE((std::is_same<squared_norm_type_t<std::vector<int>>, uint64_t>::value));
        EXPECT_TRUE((std::is_same<squared_norm_type_t<std::vector<double>>, double>::value));
        EXPECT_TRUE((std::is_same<squared_norm_type_t<std::vector<long double>>, long double>::value));
    }
}
