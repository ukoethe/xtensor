/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ENABLE_ASSERT
#define XTENSOR_ENABLE_ASSERT
#endif

#include <numeric>
#include <limits>
#include <iostream>

#include "gtest/gtest.h"
#include "xtensor/xtiny.hpp"

template <class VECTOR, class VALUE>
bool equalValue(VECTOR const & v, VALUE const & vv)
{
    for (unsigned int i = 0; i<v.size(); ++i)
        if (v[i] != vv)
            return false;
    return true;
}

template <class VECTOR1, class VECTOR2>
bool equalVector(VECTOR1 const & v1, VECTOR2 const & v2)
{
    for (unsigned int i = 0; i<v1.size(); ++i)
        if (v1[i] != v2[i])
            return false;
    return true;
}

template <class ITER1, class ITER2>
bool equalIter(ITER1 i1, ITER1 i1end, ITER2 i2, xt::index_t size)
{
    if (i1end - i1 != size)
        return false;
    for (; i1<i1end; ++i1, ++i2)
        if (*i1 != *i2)
            return false;
    return true;
}


namespace xt
{
    static const int SIZE = 3;
    using BV = tiny_array<unsigned char, SIZE>;
    using IV = tiny_array<int, SIZE>;
    using FV = tiny_array<float, SIZE>;

    static float di[] = { 1, 2, 4};
    static float df[] = { 1.2f, 2.4f, 3.6f};
    BV bv0, bv1{1}, bv3(di);
    IV iv0, iv1{1}, iv3(di);
    FV fv0, fv1{1.0f}, fv3(df);

    TEST(xtiny, traits)
    {
        EXPECT_TRUE(BV::may_use_uninitialized_memory);
        EXPECT_TRUE((tiny_array<BV, runtime_size>::may_use_uninitialized_memory));
        EXPECT_FALSE((tiny_array<tiny_array<int, runtime_size>, SIZE>::may_use_uninitialized_memory));

        EXPECT_TRUE((std::is_same<IV, promote_t<BV>>::value));
        EXPECT_TRUE((std::is_same<tiny_array<double, 3>, real_promote_t<IV>>::value));
        EXPECT_TRUE((std::is_same<typename IV::template as_type<double>, real_promote_t<IV>>::value));

        EXPECT_TRUE((std::is_same<uint64_t, squared_norm_t<tiny_array<int, 1> > >::value));
        EXPECT_TRUE((std::is_same<uint64_t, squared_norm_t<tiny_array<tiny_array<int, 1>, 1> > >::value));
        EXPECT_TRUE((std::is_same<uint64_t, squared_norm_t<tiny_array<tiny_array<int, 1>, 1> > >::value));
        EXPECT_TRUE((std::is_same<double, norm_t<tiny_array<int, 1> > >::value));
        EXPECT_TRUE((std::is_same<double, norm_t<tiny_array<tiny_array<int, 1>, 1> > >::value));
        EXPECT_TRUE((std::is_same<tiny_array<double, SIZE>, decltype(cos(iv3))>::value));
    }

    TEST(xtiny, construct)
    {
        EXPECT_TRUE(bv0.size() == SIZE);
        EXPECT_TRUE(iv0.size() == SIZE);
        EXPECT_TRUE(fv0.size() == SIZE);

        EXPECT_TRUE(equalValue(bv0, 0));
        EXPECT_TRUE(equalValue(iv0, 0));
        EXPECT_TRUE(equalValue(fv0, 0.0f));

        EXPECT_TRUE(equalValue(bv1, 1));
        EXPECT_TRUE(equalValue(iv1, 1));
        EXPECT_TRUE(equalValue(fv1, 1.0f));

        EXPECT_TRUE(equalIter(bv3.begin(), bv3.end(), di, SIZE));
        EXPECT_TRUE(equalIter(iv3.begin(), iv3.end(), di, SIZE));
        EXPECT_TRUE(equalIter(fv3.begin(), fv3.end(), df, SIZE));

        EXPECT_TRUE(!equalVector(bv3, fv3));
        EXPECT_TRUE(!equalVector(iv3, fv3));

        EXPECT_EQ(iv3, (IV{ 1, 2, 4 }));
        EXPECT_EQ(iv3, (IV{ 1.1, 2.2, 4.4 }));
        EXPECT_EQ(iv3, (IV({ 1, 2, 4 })));
        EXPECT_EQ(iv3, (IV({ 1.1, 2.2, 4.4 })));
        EXPECT_EQ(iv1, (IV{ 1 }));
        EXPECT_EQ(iv1, (IV{ 1.1 }));
        EXPECT_EQ(iv1, (IV({ 1 })));
        EXPECT_EQ(iv1, (IV({ 1.1 })));
        EXPECT_EQ(iv1, IV(tags::size = SIZE, 1));
        // these should not compile:
        // EXPECT_EQ(iv1, IV(1));
        // EXPECT_EQ(iv3, IV(1, 2, 4));

        BV bv(round(fv3));
        EXPECT_TRUE(equalIter(bv3.begin(), bv3.end(), bv.begin(), SIZE));
        EXPECT_TRUE(equalVector(bv3, bv));

        BV bv4(bv3.begin());
        EXPECT_TRUE(equalIter(bv3.begin(), bv3.end(), bv4.begin(), SIZE));
        EXPECT_TRUE(equalVector(bv3, bv4));

        BV bv5(bv3.begin(), bv3.end(), copy_reversed);
        EXPECT_TRUE(equalIter(bv3.begin(), bv3.end(),
            std::reverse_iterator<typename BV::iterator>(bv5.end()), SIZE));

        FV fv(iv3);
        EXPECT_TRUE(equalIter(iv3.begin(), iv3.end(), fv.begin(), SIZE));
        EXPECT_TRUE(equalVector(iv3, fv));

        fv = fv3;
        EXPECT_TRUE(equalIter(fv3.begin(), fv3.end(), fv.begin(), SIZE));
        EXPECT_TRUE(equalVector(fv3, fv));

        fv = bv3;
        EXPECT_TRUE(equalIter(bv3.begin(), bv3.end(), fv.begin(), SIZE));
        EXPECT_TRUE(equalVector(bv3, fv));

        EXPECT_EQ(iv3, (iv3.template subarray<0, SIZE>()));
        EXPECT_EQ(2, (iv3.template subarray<0, 2>().size()));
        EXPECT_EQ(iv3[0], (iv3.template subarray<0, 2>()[0]));
        EXPECT_EQ(iv3[1], (iv3.template subarray<0, 2>()[1]));
        EXPECT_EQ(2, (iv3.template subarray<1, 3>().size()));
        EXPECT_EQ(iv3[1], (iv3.template subarray<1, 3>()[0]));
        EXPECT_EQ(iv3[2], (iv3.template subarray<1, 3>()[1]));
        EXPECT_EQ(1, (iv3.template subarray<1, 2>().size()));
        EXPECT_EQ(iv3[1], (iv3.template subarray<1, 2>()[0]));
        EXPECT_EQ(1, (iv3.subarray(1, 2).size()));
        EXPECT_EQ(iv3[1], (iv3.subarray(1, 2)[0]));

        for (int k = 0; k<SIZE; ++k)
        {
            IV iv = IV::unit_vector(k);
            EXPECT_EQ(iv[k], 1);
            iv[k] = 0;
            EXPECT_TRUE(!any(iv));
        }

        IV seq = IV::linear_sequence(), seq_ref(tags::size = seq.size());
        std::iota(seq_ref.begin(), seq_ref.end(), 0);
        EXPECT_EQ(seq, seq_ref);

        seq = IV::linear_sequence(2);
        std::iota(seq_ref.begin(), seq_ref.end(), 2);
        EXPECT_EQ(seq, seq_ref);
        EXPECT_EQ(seq, IV::range((int)seq.size() + 2));

        seq = IV::linear_sequence(20, -1);
        std::iota(seq_ref.rbegin(), seq_ref.rend(), 20 - (int)seq.size() + 1);
        EXPECT_EQ(seq, seq_ref);

        IV r = reversed(iv3);
        for (int k = 0; k<SIZE; ++k)
            EXPECT_EQ(iv3[k], r[SIZE - 1 - k]);

        EXPECT_EQ(transpose(r, IV::linear_sequence(SIZE - 1, -1)), iv3);

        r.reverse();
        EXPECT_EQ(r, iv3);

        typedef tiny_array<typename FV::value_type, SIZE - 1> FV1;
        FV1 fv10(fv3.begin());
        EXPECT_EQ(fv10, fv3.erase(SIZE - 1));
        EXPECT_EQ(fv3, fv10.insert(SIZE - 1, fv3[SIZE - 1]));
        FV1 fv11(fv3.begin() + 1);
        EXPECT_EQ(fv11, fv3.erase(0));
    }

    TEST(xtiny, comparison)
    {
        EXPECT_TRUE(bv0 == bv0);
        EXPECT_TRUE(bv0 == 0);
        EXPECT_TRUE(0 == bv0);
        EXPECT_TRUE(iv0 == iv0);
        EXPECT_TRUE(fv0 == fv0);
        EXPECT_TRUE(fv0 == 0);
        EXPECT_TRUE(0 == fv0);
        EXPECT_TRUE(iv0 == bv0);
        EXPECT_TRUE(iv0 == fv0);
        EXPECT_TRUE(fv0 == bv0);

        EXPECT_TRUE(bv3 == bv3);
        EXPECT_TRUE(iv3 == iv3);
        EXPECT_TRUE(fv3 == fv3);
        EXPECT_TRUE(iv3 == bv3);
        EXPECT_TRUE(iv3 != fv3);
        EXPECT_TRUE(iv3 != 0);
        EXPECT_TRUE(0 != iv3);
        EXPECT_TRUE(fv3 != bv3);
        EXPECT_TRUE(fv3 != 0);
        EXPECT_TRUE(0 != fv3);

        EXPECT_TRUE(bv0 < bv1);

        EXPECT_TRUE(all_less(bv0, bv1));
        EXPECT_TRUE(!all_less(bv1, bv3));
        EXPECT_TRUE(all_greater(bv1, bv0));
        EXPECT_TRUE(!all_greater(bv3, bv1));
        EXPECT_TRUE(all_less_equal(bv0, bv1));
        EXPECT_TRUE(all_less_equal(0, bv0));
        EXPECT_TRUE(all_less_equal(bv0, 0));
        EXPECT_TRUE(all_less_equal(bv1, bv3));
        EXPECT_TRUE(!all_less_equal(bv3, bv1));
        EXPECT_TRUE(all_greater_equal(bv1, bv0));
        EXPECT_TRUE(all_greater_equal(bv3, bv1));
        EXPECT_TRUE(!all_greater_equal(bv1, bv3));

        EXPECT_TRUE(isclose(fv3, fv3));

        EXPECT_TRUE(!any(bv0) && !all(bv0) && any(bv1) && all(bv1));
        EXPECT_TRUE(!any(iv0) && !all(iv0) && any(iv1) && all(iv1));
        EXPECT_TRUE(!any(fv0) && !all(fv0) && any(fv1) && all(fv1));
        IV iv;
        iv = IV(); iv[0] = 1;
        EXPECT_TRUE(any(iv) && !all(iv));
        iv = IV(); iv[1] = 1;
        EXPECT_TRUE(any(iv) && !all(iv));
        iv = IV(); iv[SIZE - 1] = 1;
        EXPECT_TRUE(any(iv) && !all(iv));
        iv = IV{ 1 }; iv[0] = 0;
        EXPECT_TRUE(any(iv) && !all(iv));
        iv = IV{ 1 }; iv[1] = 0;
        EXPECT_TRUE(any(iv) && !all(iv));
        iv = IV{ 1 }; iv[SIZE - 1] = 0;
        EXPECT_TRUE(any(iv) && !all(iv));
    }

    TEST(xtiny, arithmetic)
    {
        using namespace cmath;

        IV ivm3 = -iv3;
        FV fvm3 = -fv3;

        int mi[] = { -1, -2, -4, -5, -8, -10 };
        float mf[] = { -1.2f, -2.4f, -3.6f, -4.8f, -8.1f, -9.7f };

        EXPECT_TRUE(equalIter(ivm3.begin(), ivm3.end(), mi, SIZE));
        EXPECT_TRUE(equalIter(fvm3.begin(), fvm3.end(), mf, SIZE));

        IV iva3 = abs(ivm3);
        FV fva3 = abs(fvm3);
        EXPECT_TRUE(equalVector(iv3, iva3));
        EXPECT_TRUE(equalVector(fv3, fva3));

        int fmi[] = { -2, -3, -4, -5, -9, -10 };
        int fpi[] = { 1, 2, 3, 4, 8, 9 };
        int ri[] = { 1, 2, 4, 5, 8, 10 };
        IV ivi3 = floor(fvm3);
        EXPECT_TRUE(equalIter(ivi3.begin(), ivi3.end(), fmi, SIZE));
        ivi3 = -ceil(fv3);
        EXPECT_TRUE(equalIter(ivi3.begin(), ivi3.end(), fmi, SIZE));
        ivi3 = round(fv3);
        EXPECT_TRUE(equalIter(ivi3.begin(), ivi3.end(), ri, SIZE));
        ivi3 = floor(fv3);
        EXPECT_TRUE(equalIter(ivi3.begin(), ivi3.end(), fpi, SIZE));
        ivi3 = -ceil(fvm3);
        EXPECT_TRUE(equalIter(ivi3.begin(), ivi3.end(), fpi, SIZE));
        ivi3 = -round(fvm3);
        EXPECT_TRUE(equalIter(ivi3.begin(), ivi3.end(), ri, SIZE));

        EXPECT_EQ(clip_lower(iv3), iv3);
        EXPECT_EQ(clip_lower(iv3, 11), IV{ 11 });
        EXPECT_EQ(clip_upper(iv3, 0), IV{ 0 });
        EXPECT_EQ(clip_upper(iv3, 11), iv3);
        EXPECT_EQ(clip(iv3, 0, 11), iv3);
        EXPECT_EQ(clip(iv3, 11, 12), IV{ 11 });
        EXPECT_EQ(clip(iv3, -1, 0), IV{ 0 });
        EXPECT_EQ(clip(iv3, IV{0 }, IV{11}), iv3);
        EXPECT_EQ(clip(iv3, IV{11}, IV{12}), IV{11});
        EXPECT_EQ(clip(iv3, IV{-1}, IV{0 }), IV{0 });

        EXPECT_TRUE(squared_norm(bv1) == SIZE);
        EXPECT_TRUE(squared_norm(iv1) == SIZE);
        EXPECT_TRUE(squared_norm(fv1) == (float)SIZE);

        float expectedSM = 1.2f*1.2f + 2.4f*2.4f + 3.6f*3.6f;
        EXPECT_NEAR(squared_norm(fv3), expectedSM, 1e-6);

        EXPECT_EQ(static_cast<uint64_t>(dot(bv3, bv3)), squared_norm(bv3));
        EXPECT_EQ(static_cast<uint64_t>(dot(iv3, bv3)), squared_norm(iv3));
        EXPECT_NEAR(dot(fv3, fv3), squared_norm(fv3), 1e-6);

        tiny_array<IV, 3> ivv{ iv3, iv3, iv3 };
        EXPECT_EQ(squared_norm(ivv), 3 * squared_norm(iv3));
        EXPECT_EQ(norm(ivv), sqrt(3.0*static_cast<double>(squared_norm(iv3))));
        EXPECT_EQ(elementwise_norm(iv3), iv3);
        EXPECT_EQ(elementwise_squared_norm(iv3), (IV{ 1, 4, 16 }));

        EXPECT_TRUE(isclose(sqrt(dot(bv3, bv3)), norm(bv3), 0.0));
        EXPECT_TRUE(isclose(sqrt(dot(iv3, bv3)), norm(iv3), 0.0));
        EXPECT_TRUE(isclose(sqrt(dot(fv3, fv3)), norm(fv3), 0.0));
        EXPECT_NEAR(sqrt(dot(bv3, bv3)), norm(bv3), 1e-6);
        EXPECT_NEAR(sqrt(dot(iv3, bv3)), norm(iv3), 1e-6);
        EXPECT_NEAR(sqrt(dot(fv3, fv3)), norm(fv3), 1e-6);

        BV bv = bv3;
        bv[2] = 200;
        uint64_t expectedSM2 = 40005;
        EXPECT_EQ(static_cast<uint64_t>(dot(bv, bv)), expectedSM2);
        EXPECT_EQ(squared_norm(bv), expectedSM2);

        EXPECT_TRUE(equalVector(bv0 + 1.0, fv1));
        EXPECT_TRUE(equalVector(1.0 + bv0, fv1));
        EXPECT_TRUE(equalVector(bv1 - 1.0, fv0));
        EXPECT_TRUE(equalVector(1.0 - bv1, fv0));
        EXPECT_TRUE(equalVector(bv3 - iv3, bv0));
        EXPECT_TRUE(equalVector(fv3 - fv3, fv0));
        BV bvp = (bv3 + bv3)*0.5;
        FV fvp = (fv3 + fv3)*0.5;
        EXPECT_TRUE(equalVector(bvp, bv3));
        EXPECT_TRUE(equalVector(fvp, fv3));
        bvp = 2.0*bv3 - bv3;
        fvp = 2.0*fv3 - fv3;
        EXPECT_TRUE(equalVector(bvp, bv3));
        EXPECT_TRUE(equalVector(fvp, fv3));

        IV ivp = bv + bv;
        int ip1[] = { 2, 4, 400, 10, 16, 20 };
        EXPECT_TRUE(equalIter(ivp.begin(), ivp.end(), ip1, SIZE));
        EXPECT_TRUE(equalVector(bv0 - iv1, -iv1));

        bvp = bv3 / 2.0;
        fvp = bv3 / 2.0;
        int ip[] = { 0, 1, 2, 3, 4, 5 };
        float fp[] = { 0.5, 1.0, 2.0, 2.5, 4.0, 5.0 };
        EXPECT_TRUE(equalIter(bvp.begin(), bvp.end(), ip, SIZE));
        EXPECT_TRUE(equalIter(fvp.begin(), fvp.end(), fp, SIZE));
        fvp = fv3 / 2.0;
        float fp1[] = { 0.6f, 1.2f, 1.8f, 2.4f, 4.05f, 4.85f };
        EXPECT_TRUE(equalIter(fvp.begin(), fvp.end(), fp1, SIZE));
        EXPECT_EQ(2.0 / fv1, 2.0 * fv1);
        float fp2[] = { 1.0f, 0.5f, 0.25f, 0.2f, 0.125f, 0.1f };
        fvp = 1.0 / bv3;
        EXPECT_TRUE(equalIter(fvp.begin(), fvp.end(), fp2, SIZE));

        int ivsq[] = { 1, 4, 16, 25, 64, 100 };
        ivp = iv3*iv3;
        EXPECT_TRUE(equalIter(ivp.begin(), ivp.end(), ivsq, SIZE));
        EXPECT_EQ(iv3 * iv1, iv3);
        EXPECT_EQ(iv0 * iv3, iv0);
        EXPECT_EQ(iv3 / iv3, iv1);
        EXPECT_EQ(iv3 % iv3, iv0);
        EXPECT_EQ(iv3 % (iv3 + iv1), iv3);

        float minRef[] = { 1.0f, 2.0f, 3.6f, 4.8f, 8.0f, 9.7f };
        float minRefScalar[] = { 1.2f, 2.4f, 3.6f, 4.0f, 4.0f, 4.0f };
        auto minRes = min(iv3, fv3);
        EXPECT_TRUE(equalIter(minRef, minRef + SIZE, minRes.cbegin(), SIZE));
        minRes = min(4.0f, fv3);
        EXPECT_TRUE(equalIter(minRefScalar, minRefScalar + SIZE, minRes.cbegin(), SIZE));
        minRes = min(fv3, 4.0f);
        EXPECT_TRUE(equalIter(minRefScalar, minRefScalar + SIZE, minRes.cbegin(), SIZE));
        IV ivmin = floor(fv3);
        ivmin[1] = 3;
        int minRef2[] = { 1, 2, 3, 4, 8, 9 };
        auto minRes2 = min(iv3, ivmin);
        EXPECT_TRUE(equalIter(minRef2, minRef2 + SIZE, minRes2.cbegin(), SIZE));
        EXPECT_EQ(min(iv3), di[0]);
        EXPECT_EQ(min(fv3), df[0]);
        EXPECT_EQ(max(iv3), di[SIZE - 1]);
        EXPECT_EQ(max(fv3), df[SIZE - 1]);

        float maxRef[] = { 1.2f, 2.4f, 4.0f, 5.0f, 8.1f, 10.0f };
        EXPECT_TRUE(equalIter(maxRef, maxRef + SIZE, max(iv3, fv3).begin(), SIZE));
        float maxRefScalar[] = { 4.0f, 4.0f, 4.0f, 4.8f, 8.1f, 9.7f };
        EXPECT_TRUE(equalIter(maxRefScalar, maxRefScalar + SIZE, max(4.0f, fv3).begin(), SIZE));
        EXPECT_TRUE(equalIter(maxRefScalar, maxRefScalar + SIZE, max(fv3, 4.0f).begin(), SIZE));
        IV ivmax = floor(fv3);
        ivmax[1] = 3;
        int maxRef2[] = { 1, 3, 4, 5, 8, 10 };
        EXPECT_TRUE(equalIter(maxRef2, maxRef2 + SIZE, max(iv3, ivmax).begin(), SIZE));

        EXPECT_EQ(sqrt(iv3 * iv3), iv3);
        EXPECT_EQ(sqrt(pow(iv3, 2)), iv3);

        EXPECT_EQ(sum(iv3),  7);
        EXPECT_EQ(sum(fv3),  7.2f);
        EXPECT_EQ(prod(iv3), 8);
        EXPECT_EQ(prod(fv3), 10.368f);
        EXPECT_NEAR(mean(iv3), 7.0 / SIZE, 1e-7);

        float cumsumRef[] = { 1.2f, 3.6f, 7.2f};
        FV cs = cumsum(fv3), csr(cumsumRef);
        EXPECT_TRUE(isclose(cs, csr, 1e-6f));
        float cumprodRef[] = { 1.2f, 2.88f, 10.368f};
        FV cr = cumprod(fv3), crr(cumprodRef);
        EXPECT_TRUE(isclose(cr, crr, 1e-6f));

        tiny_array<int, 4> src{ 1, 2, -3, -4 }, signs{ 2, -3, 4, -5 };
        EXPECT_EQ(copysign(src, signs), (tiny_array<int, 4>{1, -2, 3, -4}));

        tiny_array<double, 3> left{ 3., 5., 8. }, right{ 4., 12., 15. };
        EXPECT_EQ(hypot(left, right), (tiny_array<double, 3>{5., 13., 17.}));

        int oddRef[] = { 1, 0, 0, 1, 0, 0 };
        EXPECT_TRUE(equalIter(oddRef, oddRef + SIZE, odd(iv3).begin(), SIZE));
        EXPECT_TRUE(equalIter(oddRef, oddRef + SIZE, (iv3 & 1).begin(), SIZE));
    }

    TEST(xtiny, cross_product)
    {
        EXPECT_EQ(cross(bv3, bv3), IV{0});
        EXPECT_EQ(cross(iv3, bv3), IV{0});
        EXPECT_TRUE(isclose(cross(fv3, fv3), FV{ 0.0f }, 1e-6f));

        FV cr = cross(fv1, fv3), crr{ 1.2f, -2.4f, 1.2f };
        EXPECT_TRUE(isclose(cr, crr, 1e-6f));
    }

    TEST(xtiny, ostream)
    {
        std::ostringstream out;
        out << iv3;
        std::string expected("{1, 2, 4}");
        EXPECT_EQ(expected, out.str());
        out << "Testing.." << fv3 << 42;
        out << bv3 << std::endl;
    }

    TEST(xtiny, 2D)
    {
        using std::sqrt;

        using Array = tiny_array<int, 2, 3>;
        using Index = tiny_array<index_t, 2>;

        EXPECT_TRUE(Array::static_ndim == 2);
        EXPECT_TRUE(Array::static_size == 6);
        EXPECT_TRUE((std::is_same<Index, Array::index_type>::value));

        int adata[] = { 4,5,6,7,8,9 };
        Array a(adata);
        EXPECT_EQ(a.ndim(), 2);
        EXPECT_EQ(a.size(), 6);
        EXPECT_EQ(a.shape(), (Index{ 2, 3 }));

        int count = 0, i, j;
        Index idx;
        for (i = 0, idx[0] = 0; i<2; ++i, ++idx[0])
        {
            for (j = 0, idx[1] = 0; j<3; ++j, ++count, ++idx[1])
            {
                EXPECT_EQ(a[count], adata[count]);
                EXPECT_EQ((a[{i, j}]), adata[count]);
                EXPECT_EQ(a[idx], adata[count]);
                EXPECT_EQ(a(i, j), adata[count]);
            }
        }
        {
            std::string s = "{4, 5, 6,\n 7, 8, 9}";
            std::stringstream ss;
            ss << a;
            EXPECT_EQ(s, ss.str());
        }

        Array::as_type<float> b = a;
        EXPECT_EQ(a, b);

        int adata2[] = { 0,1,2,3,4,5 };
        a = { 0,1,2,3,4,5 };
        EXPECT_TRUE(equalIter(a.begin(), a.end(), adata2, a.size()));
        Array c = reversed(a);
        EXPECT_TRUE(equalIter(c.rbegin(), c.rend(), adata2, c.size()));

        EXPECT_TRUE(a == a);
        EXPECT_TRUE(a != b);
        EXPECT_TRUE(a < b);
        EXPECT_TRUE(any(a));
        EXPECT_TRUE(!all(a));
        EXPECT_TRUE(any(b));
        EXPECT_TRUE(all(b));
        EXPECT_TRUE(!all_zero(a));
        EXPECT_TRUE(all_less(a, b));
        EXPECT_TRUE(all_less_equal(a, b));
        EXPECT_TRUE(!all_greater(a, b));
        EXPECT_TRUE(!all_greater_equal(a, b));
        EXPECT_TRUE(isclose(a, b, 10.0f));

        EXPECT_EQ(squared_norm(a), 55u);
        EXPECT_TRUE(isclose(norm(a), sqrt(55.0), 1e-15));
        EXPECT_NEAR(norm(a), sqrt(55.0), 1e-15);
        EXPECT_EQ(min(a), 0);
        EXPECT_EQ(max(a), 5);
        EXPECT_EQ(max(a, b), b);

        swap(b, c);
        EXPECT_TRUE(equalIter(c.cbegin(), c.cend(), adata, c.size()));
        EXPECT_TRUE(equalIter(b.crbegin(), b.crend(), adata2, b.size()));

        int eyedata[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        auto eye = Array::eye<3>();
        EXPECT_TRUE(equalIter(eye.begin(), eye.end(), eyedata, eye.size()));
    }

    TEST(xtiny, runtime_size)
    {
        using A = tiny_array<int>;
        using V1 = tiny_array<int, 1>;

        EXPECT_TRUE(typeid(A) == typeid(tiny_array<int, runtime_size>));

        A a{ 1,2,3 }, b{ 1,2,3 }, c = a, d = a + b, e(3);
        EXPECT_EQ(a.size(), 3);
        EXPECT_EQ(b.size(), 3);
        EXPECT_EQ(c.size(), 3);
        EXPECT_EQ(d.size(), 3);
        EXPECT_EQ(e.size(), 3);
        EXPECT_EQ(a, b);
        EXPECT_EQ(a, c);
        EXPECT_TRUE(a != d);
        EXPECT_TRUE(a != e);
        EXPECT_TRUE(a < d);
        EXPECT_TRUE(e < a);
        EXPECT_EQ(e, (A{ 0,0,0 }));

        EXPECT_EQ(iv3, (A{ 1, 2, 4 }));
        EXPECT_EQ(iv3, (A{ 1.1, 2.2, 4.4 }));
        EXPECT_EQ(iv3, (A({ 1, 2, 4 })));
        EXPECT_EQ(iv3, (A({ 1.1, 2.2, 4.4 })));
        EXPECT_EQ(V1{1}, (A{ 1 }));
        EXPECT_EQ(V1{1}, (A{ 1.1 }));
        EXPECT_EQ(V1{1}, (A({ 1 })));
        EXPECT_EQ(V1{1}, (A({ 1.1 })));
        EXPECT_EQ(iv0, A(SIZE));
        EXPECT_EQ(iv0, A(tags::size = SIZE));
        EXPECT_EQ(iv1, A(SIZE, 1));
        EXPECT_EQ(iv1, A(tags::size = SIZE, 1));

        c.init(2, 4, 6);
        EXPECT_EQ(d, c);
        c.init({ 1,2,3 });
        EXPECT_EQ(a, c);
        c = 2 * a;
        EXPECT_EQ(d, c);
        c.reverse();
        EXPECT_EQ(c, (A{ 6,4,2 }));
        EXPECT_EQ(c, reversed(d));
        c = c - 2;
        EXPECT_TRUE(all(d));
        EXPECT_TRUE(!all(c));
        EXPECT_TRUE(any(c));
        EXPECT_TRUE(!all_zero(c));
        EXPECT_TRUE(!all(e));
        EXPECT_TRUE(!any(e));
        EXPECT_TRUE(all_zero(e));

        EXPECT_EQ(prod(a), 6);
        EXPECT_EQ(prod(A()), 0);

        EXPECT_EQ(cross(a, a), e);
        EXPECT_EQ(dot(a, a), 14);
        EXPECT_EQ(squared_norm(a), 14u);

        EXPECT_EQ(a.erase(1), (A{ 1,3 }));
        EXPECT_EQ(a.insert(3, 4), (A{ 1,2,3,4 }));

        // testing move constructor and assignment
        EXPECT_EQ(std::move(A{ 1,2,3 }), (A{ 1,2,3 }));
        EXPECT_EQ(A(a.insert(3, 4)), (A{ 1,2,3,4 }));
        a = a.insert(3, 4);
        EXPECT_EQ(a, (A{ 1,2,3,4 }));

        A r = A::range(2, 6);
        EXPECT_EQ(r, (A{ 2,3,4,5 }));
        EXPECT_EQ(r.subarray(1, 3).size(), 2);
        EXPECT_EQ(r.subarray(1, 3), (A{ 3,4 }));
        EXPECT_EQ((r.template subarray<1, 3>().size()), 2);
        EXPECT_EQ((r.template subarray<1, 3>()), (A{ 3,4 }));

        EXPECT_EQ(A::range(0, 6, 3), (A{ 0,3 }));
        EXPECT_EQ(A::range(0, 7, 3), (A{ 0,3,6 }));
        EXPECT_EQ(A::range(0, 8, 3), (A{ 0,3,6 }));
        EXPECT_EQ(A::range(0, 9, 3), (A{ 0,3,6 }));
        EXPECT_EQ(A::range(10, 2, -2), (A{ 10, 8, 6, 4 }));
        EXPECT_EQ(A::range(10, 1, -2), (A{ 10, 8, 6, 4, 2 }));
        EXPECT_EQ(A::range(10, 0, -2), (A{ 10, 8, 6, 4, 2 }));

        EXPECT_EQ(transpose(A::range(1, 4)), (A{ 3,2,1 }));
        EXPECT_EQ(transpose(A::range(1, 4), A{ 1,2,0 }), (A{ 2,3,1 }));

        EXPECT_THROW(A(3) / A(2), std::runtime_error);

        using TA = tiny_array<int, 3>;
        TA s(A{ 1,2,3 });
        EXPECT_EQ(s, (TA{ 1,2,3 }));
        s = A{ 3,4,5 };
        EXPECT_EQ(s, (TA{ 3,4,5 }));

        EXPECT_THROW({ TA(A{ 1,2,3,4 }); }, std::runtime_error);

        EXPECT_EQ((A{ 0,0,0 }), A(tags::size = 3));
        EXPECT_EQ((A{ 1,1,1 }), A(tags::size = 3, 1));

        EXPECT_EQ(A::unit_vector(tags::size = 3, 1), TA::unit_vector(1));
    }
} // namespace xt