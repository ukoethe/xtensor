/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTINY_HPP
#define XTINY_HPP

#include <type_traits>
#include <iterator>
#include "xexception.hpp"
#include "xtensor.hpp"
#include "xstrides.hpp"

namespace xt
{
    template <class T, int N>
    class xtiny_array;

    template <class T, int N>
    struct xcontainer_inner_types<xtiny_array<T, N>>
    {
        using temporary_type = xtiny_array<T, N>;
    };

    template <class T, int N>
    struct xiterable_inner_types<xtiny_array<T, N>>
    {
        using inner_shape_type = std::array<size_t, 1>;
        using stepper = xindexed_stepper<xtiny_array<T, N>, false>;
        using const_stepper = xindexed_stepper<xtiny_array<T, N>, true>;
    };

    template <class T, int N>
    class xtiny_array
    : public xiterable<xtiny_array<T, N>>
    , public xcontainer_semantic<xtiny_array<T, N>>
    {

      public:
        using value_type             = T;
        using const_value_type       = typename std::add_const<T>::type;
        using reference              = value_type &;
        using const_reference        = const_value_type &;
        using pointer                = value_type *;
        using const_pointer          = const_value_type *;
        using iterator               = value_type *;
        using const_iterator         = const_value_type *;
        using reverse_iterator       = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        using size_type              = size_t;
        using difference_type        = std::ptrdiff_t;
        using shape_type             = std::array<size_t, 1>;
        using strides_type           = shape_type;
        using container_type         = T[N];
        static constexpr layout_type static_layout = layout_type::row_major;
        static constexpr bool contiguous_layout = true;

        using self_type = xtiny_array<T, N>;
        using semantic_base = xcontainer_semantic<self_type>;

        using inner_shape_type = shape_type;
        using inner_strides_type = inner_shape_type;

        using iterable_base = xiterable<self_type>;
        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;


        T data_[N];

        xtiny_array(const_reference initial = value_type())
        {
            for(int k=0; k<N; ++k)
                data_[k] = initial;
        }

        xtiny_array(const xtiny_array&) = default;
        xtiny_array& operator=(const xtiny_array&) = default;

        xtiny_array(xtiny_array&&) = default;
        xtiny_array& operator=(xtiny_array&&) = default;

        template <class E>
        xtiny_array(const xexpression<E>& e)
        {
            semantic_base::assign(e);
        }

        template <class E>
        self_type& operator=(const xexpression<E>& e)
        {
            return semantic_base::operator=(e);
        }

        reference operator()(size_type i)
        {
            return data_[i];
        }

        const_reference operator()(size_type i) const
        {
            return data_[i];
        }

        reference operator()()
        {
            return data_[0];
        }

        const_reference operator()() const
        {
            return data_[0];
        }

        reference operator[](size_type i)
        {
            return data_[i];
        }

        const_reference operator[](size_type i) const
        {
            return data_[i];
        }

        reference operator[](const xindex& index)
        {
            return element(index.cbegin(), index.cend());
        }

        const_reference operator[](const xindex& index) const
        {
            return element(index.cbegin(), index.cend());
        }

        template <class It>
        reference element(It first, It last)
        {
            return data_[*first];
        }

        template <class It>
        const_reference element(It first, It last) const
        {
            return data_[*first];
        }

        constexpr size_type dimension() const
        {
            return 1;
        }

        // constexpr size_type size() const
        // {
            // return N;
        // }

        const shape_type & shape() const // must return by reference
        {
            static constexpr shape_type s{N};
            return s;
        }

        shape_type strides() const
        {
            return {1};
        }

        template <class S>
        bool broadcast_shape(S& s) const
        {
            return xt::broadcast_shape(shape(), s);
        }

        template <class S>
        bool is_trivial_broadcast(const S& str) const noexcept
        {
            return false;
        }

        void reshape(const shape_type& s)
        {
            XTENSOR_PRECONDITION(s == shape(),
                "xtiny_array::reshape(): invalid target shape.");
        }

        // void reshape(const shape_type& shape, layout_type l)
        // {
            // XTENSOR_PRECONDITION(s == shape(),
                // "xtiny_array::reshape(): invalid target shape.");
        // }

        // void reshape(const shape_type& shape, const strides_type& strides)
        // {
            // XTENSOR_PRECONDITION(s == shape(),
                // "xtiny_array::reshape(): invalid target shape.");
        // }

        template <class ST>
        stepper stepper_begin(const ST& s)
        {
            size_type offset = s.size() - dimension();
            return stepper(this, offset);
        }

        template <class ST>
        stepper stepper_end(const ST& s, layout_type = layout_type::row_major)
        {
            size_type offset = s.size() - dimension();
            return stepper(this, offset, true);
        }

        template <class ST>
        const_stepper stepper_begin(const ST& s) const
        {
            size_type offset = s.size() - dimension();
            return const_stepper(this, offset);
        }

        template <class ST>
        const_stepper stepper_end(const ST& s, layout_type = layout_type::row_major) const
        {
            size_type offset = s.size() - dimension();
            return const_stepper(this, offset, true);
        }

        // reference front()
        // {
            // return data_[0];
        // }

        // const_reference front() const
        // {
            // return data_[0];
        // }

        // reference back()
        // {
            // return data_[N-1];
        // }

        // const_reference back() const
        // {
            // return data_[N-1];
        // }

        // pointer data()
        // {
            // return data_;
        // }

        // const_pointer data() const
        // {
            // return data_;
        // }

        // iterator begin()
        // {
            // return data_;
        // }

        // const_iterator begin() const
        // {
            // return data_;
        // }

        // const_iterator cbegin() const
        // {
            // return data_;
        // }

        // iterator end()
        // {
            // return data_ + N;
        // }

        // const_iterator end() const
        // {
            // return data_ + N;
        // }

        // const_iterator cend() const
        // {
            // return data_ + N;
        // }
    };
} // namespace xt

#endif // XTINY_HPP
