/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "test_common.hpp"

namespace xt
{
    using array_type = xarray<double>;
    using tensor_type = xtensor<double, 2>;
    using array_shape = array_type::shape_type;
    using tensor_shape = tensor_type::shape_type;

    TEST(xtensor_semantic, tensor_plus_tensor)
    {
        tensor_shape s = {3, 2};
        tensor_type t1(s, 3.2);
        tensor_type t2(s, 2.5);
        tensor_type res = t1 + t2;
        EXPECT_EQ(res(0, 0), t1(0, 0) + t2(0, 0));
    }

    TEST(xtensor_semantic, tensor_plus_array)
    {
        tensor_shape s1 = {3, 2};
        tensor_type t1(s1, 3.2);
        array_shape s2 = {3, 2};
        array_type t2(s2, 2.5);
        tensor_type res = t1 + t2;
        EXPECT_EQ(res(0, 0), t1(0, 0) + t2(0, 0));
    }

    TEST(xtensor_semantic, array_plus_tensor)
    {
        tensor_shape s1 = {3, 2};
        tensor_type t1(s1, 3.2);
        array_shape s2 = {3, 2};
        array_type t2(s2, 2.5);
        array_type res = t1 + t2;
        EXPECT_EQ(res(0, 0), t1(0, 0) + t2(0, 0));
    }
}

