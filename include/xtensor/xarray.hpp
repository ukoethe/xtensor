/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XARRAY_HPP
#define XARRAY_HPP

#include <initializer_list>
#include <utility>
#include <vector>
#include <algorithm>

#include "xarray_base.hpp"
#include "xsemantic.hpp"

namespace xt
{

    /**********************
     * xarray declaration *
     **********************/

    template <class T>
    class xarray;

    template <class T>
    struct array_inner_types<xarray<T>>
    {
        using container_type = std::vector<T>;
        using temporary_type = xarray<T>;
    };

    template <class T>
    class xarray : public xarray_base<xarray<T>>,
                   public xarray_semantic<xarray<T>>
    {

    public:

        using self_type = xarray<T>;
        using base_type = xarray_base<self_type>;
        using semantic_base = xarray_semantic<self_type>;
        using container_type = typename base_type::container_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;

        using closure_type = const self_type&;

        xarray();
        explicit xarray(const shape_type& shape, layout l = layout::row_major);
        explicit xarray(const shape_type& shape, const_reference value, layout l = layout::row_major);
        explicit xarray(const shape_type& shape, const strides_type& strides);
        explicit xarray(const shape_type& shape, const strides_type& strides, const_reference value);

        explicit xarray(const T& t);
        xarray(std::initializer_list<T> t);
        xarray(std::initializer_list<std::initializer_list<T>> t);
        xarray(std::initializer_list<std::initializer_list<std::initializer_list<T>>> t);
        xarray(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> t);
        xarray(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>>> t);

        ~xarray() = default;

        xarray(const xarray&) = default;
        xarray& operator=(const xarray&) = default;

        xarray(xarray&&) = default;
        xarray& operator=(xarray&&) = default;

        template <class E>
        xarray(const xexpression<E>& e);

        template <class E>
        xarray& operator=(const xexpression<E>& e);

    private:

        container_type m_data;

        container_type& data_impl();
        const container_type& data_impl() const;

        friend class xarray_base<xarray<T>>;
    };

    /******************************
     * xarray_adaptor declaration *
     ******************************/

    template <class C>
    class xarray_adaptor;

    template <class C>
    struct array_inner_types<xarray_adaptor<C>>
    {
        using container_type = C;
        using temporary_type = xarray<typename C::value_type>;
    };

    template <class C>
    class xarray_adaptor : public xarray_base<xarray_adaptor<C>>,
                           public xadaptor_semantic<xarray_adaptor<C>>
    {

    public:

        using self_type = xarray_adaptor<C>;
        using base_type = xarray_base<self_type>;
        using semantic_base = xadaptor_semantic<self_type>;
        using container_type = typename base_type::container_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;

        using closure_type = const self_type&;

        xarray_adaptor(container_type& data);
        xarray_adaptor(container_type& data, const shape_type& shape, layout l = layout::row_major);
        xarray_adaptor(container_type& data, const shape_type& shape, const strides_type& strides);

        ~xarray_adaptor() = default;

        xarray_adaptor(const xarray_adaptor&) = default;
        xarray_adaptor& operator=(const xarray_adaptor&);

        xarray_adaptor(xarray_adaptor&&) = default;
        xarray_adaptor& operator=(xarray_adaptor&&);

        template <class E>
        xarray_adaptor& operator=(const xexpression<E>& e);

    private:

        container_type& m_data;

        container_type& data_impl();
        const container_type& data_impl() const;

        using temporary_type = typename array_inner_types<self_type>::temporary_type;
        void assign_temporary_impl(temporary_type& tmp);

        friend class xarray_base<xarray_adaptor<C>>;
        friend class xadaptor_semantic<xarray_adaptor<C>>;
    };

    /*************************
     * xarray implementation *
     *************************/

    template <class T>
    inline xarray<T>::xarray()
        : base_type(), m_data(1, value_type())
    {
    }

    template <class T>
    inline xarray<T>::xarray(const shape_type& shape, layout l)
        : base_type()
    {
        base_type::reshape(shape, l);
    }

    template <class T>
    inline xarray<T>::xarray(const shape_type& shape, const_reference value, layout l)
        : base_type()
    {
        base_type::reshape(shape, l);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    template <class T>
    inline xarray<T>::xarray(const shape_type& shape, const strides_type& strides)
        : base_type()
    {
        base_type::reshape(shape, strides);
    }

    template <class T>
    inline xarray<T>::xarray(const shape_type& shape, const strides_type& strides, const_reference value)
        : base_type()
    {
        base_type::reshape(shape, strides);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    template <class T>
    inline xarray<T>::xarray(const T& t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    template <class T>
    inline xarray<T>::xarray(std::initializer_list<T> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    template <class T>
    inline xarray<T>::xarray(std::initializer_list<std::initializer_list<T>> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    template <class T>
    inline xarray<T>::xarray(std::initializer_list<std::initializer_list<std::initializer_list<T>>> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    template <class T>
    inline xarray<T>::xarray(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    template <class T>
    inline xarray<T>::xarray(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>>> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    template <class T>
    template <class E>
    inline xarray<T>::xarray(const xexpression<E>& e)
        : base_type()
    {
        semantic_base::assign(e);
    }

    template <class T>
    template <class E>
    inline auto xarray<T>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    template <class T>
    inline auto xarray<T>::data_impl() -> container_type&
    {
        return m_data;
    }

    template <class T>
    inline auto xarray<T>::data_impl() const -> const container_type&
    {
        return m_data;
    }

    /******************
     * xarray_adaptor *
     ******************/

    template <class C>
    inline xarray_adaptor<C>::xarray_adaptor(container_type& data)
        : base_type(), m_data(data)
    {
    }

    template <class C>
    inline xarray_adaptor<C>::xarray_adaptor(container_type& data, const shape_type& shape, layout l)
        : base_type(), m_data(data)
    {
        base_type::reshape(shape, l);
    }

    template <class C>
    inline xarray_adaptor<C>::xarray_adaptor(container_type& data, const shape_type& shape, const strides_type& strides)
        : base_type(), m_data(data)
    {
        base_type::reshape(shape, strides);
    }

    template <class C>
    inline xarray_adaptor<C>& xarray_adaptor<C>::operator=(const xarray_adaptor& rhs)
    {
        base_type::operator=(rhs);
        m_data = rhs.m_data;
        return *this;
    }

    template <class C>
    inline xarray_adaptor<C>& xarray_adaptor<C>::operator=(xarray_adaptor&& rhs)
    {
        base_type::operator=(std::move(rhs));
        m_data = rhs.m_data;
        return *this;
    }

    template <class C>
    template <class E>
    inline auto xarray_adaptor<C>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    template <class C>
    inline auto xarray_adaptor<C>::data_impl() -> container_type&
    {
        return m_data;
    }

    template <class C>
    inline auto xarray_adaptor<C>::data_impl() const -> const container_type&
    {
        return m_data;
    }

    template <class C>
    inline void xarray_adaptor<C>::assign_temporary_impl(temporary_type& tmp)
    {
        // TODO (performance improvement) : consider moving tmps
        // shape and strides
        base_type::get_shape() = tmp.shape();
        base_type::get_strides() = tmp.strides();
        base_type::get_backstrides() = tmp.backstrides();
        m_data.resize(tmp.size());
        std::copy(tmp.data().begin(), tmp.data().end(), m_data.begin());
    }
}

#endif

