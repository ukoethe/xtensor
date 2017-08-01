/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_XTINY_HPP
#define XTENSOR_XTINY_HPP

#include <iosfwd>
#include <algorithm>
#include <memory>
#include <iterator>
#include <utility>
#include <tuple>  // std::ignore

#include "xtags.hpp"
#include "xconcepts.hpp"
#include "xexception.hpp"
#include "xmathutil.hpp"

#ifdef XTENSOR_CHECK_BOUNDS
    #define XTENSOR_ASSERT_INSIDE(array, diff) \
      xtensor_precondition(diff >= 0 && diff < array.size(), "Index out of bounds")
#else
    #define XTENSOR_ASSERT_INSIDE(array, diff)
#endif

namespace xt {

    /** \brief The general type of array indices.

        Note that this is a signed type, so that negative indices
        and index differences work as intuitively expected.
    */
using index_t = std::ptrdiff_t;

template <class VALUETYPE, bool owns_memory, int ... N>
class tiny_array_impl;

template <class VALUETYPE, int M=runtime_size, int ... N>
using tiny_array = tiny_array_impl<VALUETYPE, true, M, N...>;

template <class VALUETYPE, int M, int N>
using tiny_matrix = tiny_array_impl<VALUETYPE, true, M, N>;

template <class VALUETYPE, int M=runtime_size, int ... N>
using tiny_array_adaptor = tiny_array_impl<VALUETYPE, false, M, N...>;

namespace detail  {

template<class T>
struct may_use_uninitialized_memory
{
    static const bool value = std::is_scalar<T>::value || std::is_pod<T>::value;
};

template<class T, bool owns_memory, int ... N>
struct may_use_uninitialized_memory<tiny_array_impl<T, owns_memory, N...>>
{
    static const bool value = may_use_uninitialized_memory<T>::value;
};

template<class T, bool owns_memory>
struct may_use_uninitialized_memory<tiny_array_impl<T, owns_memory, runtime_size>>
{
    static const bool value = false;
};

template <index_t LEVEL, int ... N>
struct tiny_shape_helper;

template <index_t LEVEL, int N, int ... REST>
struct tiny_shape_helper<LEVEL, N, REST...>
{
    static_assert(N >= 0, "tiny_array_impl(): array must have non-negative shape.");
    using next_type = tiny_shape_helper<LEVEL+1, REST...>;

    static const index_t level      = LEVEL;
    static const index_t stride     = next_type::total_size;
    static const index_t total_size = N * stride;
    static const index_t alloc_size = total_size;

    static index_t offset(index_t const * coord)
    {
        return stride*coord[level] + next_type::offset(coord);
    }

    template <class ... V>
    static index_t offset(index_t i, V...rest)
    {
        return stride*i + next_type::offset(rest...);
    }
};

template <index_t LEVEL, int N>
struct tiny_shape_helper<LEVEL, N>
{
    static_assert(N >= 0, "tiny_array_impl(): array must have non-negative shape.");
    static const index_t level      = LEVEL;
    static const index_t stride     = 1;
    static const index_t total_size = N;
    static const index_t alloc_size = total_size;

    static index_t offset(index_t const * coord)
    {
        return coord[level];
    }

    static index_t offset(index_t i)
    {
        return i;
    }
};

template <index_t LEVEL>
struct tiny_shape_helper<LEVEL, 0>
{
    static const index_t level      = LEVEL;
    static const index_t stride     = 1;
    static const index_t total_size = 0;
    static const index_t alloc_size = 1;

    static index_t offset(index_t const * coord)
    {
        return coord[level];
    }

    static index_t offset(index_t i)
    {
        return i;
    }
};

template <int ... N>
struct tiny_size_helper
{
    static const index_t value = tiny_shape_helper<0, N...>::total_size;
    static const index_t ndim  = sizeof...(N);
};

template <int N0, int ... N>
struct tiny_array_is_static
{
    static const int ndim = sizeof...(N)+1;
    static const bool value = ndim > 1 || N0 != runtime_size;
};

} // namespace detail

#define XTENSOR_ASSERT_RUNTIME_SIZE(SHAPE, PREDICATE, MESSAGE) \
    if(detail::tiny_array_is_static<SHAPE>::value) {} else \
        xtensor_precondition(PREDICATE, MESSAGE)

/********************************************************/
/*                                                      */
/*                    tiny_array_impl                   */
/*                                                      */
/********************************************************/

/** \brief Class for fixed size arrays.
    \ingroup RangesAndPoints

    This class is typically used via the type alias \ref tiny_array or \ref tiny_matrix.

    This class encapsulates a static array of the specified VALUETYPE with
    (possibly multi-dimensional) shape given by the sequence <tt>index_t ... N</tt>.
    The interface conforms to STL vector, except that there are no functions
    that change the size of a tiny_array_impl.

    \ref TinyArrayOperators "Arithmetic operations"
    on TinyArrays are defined as component-wise applications of these
    operations.

    See also:<br>
    <UL style="list-style-image:url(documents/bullet.gif)">
        <LI> \ref xt::tiny_array
        <LI> \ref xt::tiny_array_matrix
        <LI> \ref xt::tiny_array_adaptor
        <LI> \ref TinyArrayOperators
    </UL>
**/
template <class VALUETYPE, bool OWNS_MEMORY, int ... N>
class tiny_array_impl
: public tiny_array_tag
{
  protected:
    using shape_helper = detail::tiny_shape_helper<0, N...>;

    using data_array_type = typename std::conditional<OWNS_MEMORY,
                                                VALUETYPE[shape_helper::alloc_size],
                                                VALUETYPE *>::type;

    template <int LEVEL, class ... V2>
    void init_impl(VALUETYPE v1, V2... v2)
    {
        data_[LEVEL] = v1;
        init_impl<LEVEL+1>(v2...);
    }

    template <int LEVEL>
    void init_impl(VALUETYPE v1)
    {
        data_[LEVEL] = v1;
    }

    template <class ITERATOR,
              XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value>>
    void init_impl(ITERATOR u)
    {
        for(index_t k=0; k < static_size; ++k, ++u)
            data_[k] = static_cast<VALUETYPE>(*u);
    }

  public:

    template <class NEW_VALUETYPE>
    using as_type = tiny_array_impl<NEW_VALUETYPE, OWNS_MEMORY, N...>;

    using value_type             = VALUETYPE;
    using const_value_type       = typename std::add_const<VALUETYPE>::type;
    using reference              = value_type &;
    using const_reference        = const_value_type &;
    using pointer                = value_type *;
    using const_pointer          = const_value_type *;
    using iterator               = value_type *;
    using const_iterator         = const_value_type *;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using size_type              = std::size_t;
    using difference_type        = std::ptrdiff_t;
    using index_type             = tiny_array<index_t, sizeof...(N)>;

    static const index_t static_ndim  = sizeof...(N);
    static const index_t static_size  = shape_helper::total_size;
    static const bool may_use_uninitialized_memory =
                                   detail::may_use_uninitialized_memory<VALUETYPE>::value;

    // general constructors

    constexpr tiny_array_impl(tiny_array_impl const &) = default;

    explicit tiny_array_impl(skip_initialization_tag)
    {}

        // for compatibility with tiny_array_impl<..., runtime_size>
    tiny_array_impl(tags::size_proxy const & size, skip_initialization_tag)
    {
        std::ignore = size;
        XTENSOR_ASSERT_MSG(size.value == static_size,
            "tiny_array_impl(size): size argument conflicts with array length.");
    }

        // for compatibility with tiny_array_impl<..., runtime_size>
    tiny_array_impl(index_t size, skip_initialization_tag)
    {
        std::ignore = size;
        XTENSOR_ASSERT_MSG(size == static_size,
            "tiny_array_impl(size): size argument conflicts with array length.");
    }

    // constructors when OWNS_MEMORY == false

    template <bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< !owns_memory >>
    explicit
    tiny_array_impl()
    : data_(nullptr)
    {}

    template <bool OTHER_OWNS_MEMORY,
              bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< !owns_memory >>
    explicit
    tiny_array_impl(tiny_array_impl<VALUETYPE, OTHER_OWNS_MEMORY, N...> const & other)
    : data_(other.data())
    {
        xtensor_precondition(size() == other.size(),
            "tiny_array_impl(tiny_array_impl): size mismatch.");
    }

        /** Construct view for given pointer
        */
    template <bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< !owns_memory >>
    explicit tiny_array_impl(pointer u, pointer end  = 0)
    : data_(u)
    {
        std::ignore = end;
        XTENSOR_ASSERT_MSG(end == 0 || end - u == static_size,
            "tiny_array_impl(const_pointer u, const_pointer end): size mismatch.");
    }

    // constructors when OWNS_MEMORY == true

    template <bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< owns_memory >>
    explicit
    tiny_array_impl(value_type v = value_type())
    {
        init(v);
    }

        // for compatibility with tiny_array_impl<..., runtime_size>
    template <bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< owns_memory >>
    explicit
    tiny_array_impl(tags::size_proxy const & size,
                    value_type const & v = value_type())
    : tiny_array_impl(v)
    {
        std::ignore = size;
        XTENSOR_ASSERT_MSG(size.value == static_size,
            "tiny_array_impl(size): size argument conflicts with array length.");
    }

    template <class V, bool OTHER_OWNS_MEMORY, int ... M,
              bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< owns_memory >>
    tiny_array_impl(tiny_array_impl<V, OTHER_OWNS_MEMORY, M...> const & other)
    {
        if(other.size() == 0)
        {
            init();
        }
        else
        {
            xtensor_precondition(size() == other.size(),
                "tiny_array_impl(tiny_array_impl): size mismatch.");
            init_impl(other.begin());
        }
    }

        // This constructor would allow construction with round brackets, e.g.:
        //     tiny_array<int, 1> a(2);
        // However, this may lead to bugs when fixed-size arrays are mixed with
        // runtime_size arrays, where
        //     tiny_array<int, runtime_size> a(2);
        // constructs an array of length 2 with initial value 0. To avoid such bugs,
        // construction is restricted to curly braces:
        //     tiny_array<int, 1> a{2};
        //
    // template <class ... V,
    //          bool owns_memory=OWNS_MEMORY,
    //          XTENSOR_REQUIRE< owns_memory >>
    // constexpr
    // tiny_array_impl(value_type v0, value_type v1, V ... v)
    // : data_{VALUETYPE(v0), VALUETYPE(v1), VALUETYPE(v)...}
    // {
        // static_assert(sizeof...(V)+2 == static_size,
                      // "tiny_array_impl(): number of constructor arguments contradicts size().");
    // }

    template <class V,
              bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< owns_memory >>
    tiny_array_impl(std::initializer_list<V> v)
    {
        if(v.size() == 1)
            init(static_cast<value_type>(*v.begin()));
        else if(v.size() == static_size)
            init_impl(v.begin());
        else
            xtensor_precondition(false,
                "tiny_array_impl(std::initializer_list<V>): wrong initialization size (expected: "
                + std::to_string(static_size) + ", got: " + std::to_string(v.size()) +")");
    }

    template <class U,
              bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< owns_memory >>
    explicit tiny_array_impl(U const * u, U const *  end  = 0)
    {
        std::ignore = end;
        XTENSOR_ASSERT_MSG(end == 0 || end - u == static_size,
            "tiny_array_impl(U const * u, U const * end): size mismatch.");
        init_impl(u);
    }

    template <class ITERATOR,
              bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value && owns_memory> >
    tiny_array_impl(ITERATOR u, ITERATOR end)
    {
        std::ignore = end;
        XTENSOR_ASSERT_MSG(std::distance(u, end) == static_size,
            "tiny_array_impl(ITERATOR u, ITERATOR end): size mismatch.");
        init_impl(u);
    }
    template <class ITERATOR,
              XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value> >
    tiny_array_impl(ITERATOR u, ITERATOR end, reverse_copy_tag)
    {
        XTENSOR_ASSERT_MSG(std::distance(u, end) == static_size,
            "tiny_array_impl(ITERATOR u, ITERATOR end, reverse_copy_tag): size mismatch.");
        for(int i=0; i<static_size; ++i)
        {
            --end;
            data_[i] = static_cast<value_type>(*end);
        }
    }

    template <bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< owns_memory >>
    explicit tiny_array_impl(value_type const (&u)[1])
    {
        init(*u);
    }

    template <class U,
              bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< owns_memory >>
    explicit tiny_array_impl(U const (&u)[1])
    {
        init(static_cast<value_type>(*u));
    }

    template <class U, int S=static_size,
              bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE<(S > 1) && owns_memory>>
    explicit tiny_array_impl(U const (&u)[static_size])
    {
        init_impl(u);
    }

    // assignment

    tiny_array_impl & operator=(tiny_array_impl const &) = default;

    tiny_array_impl & operator=(value_type v)
    {
        init(v);
        return *this;
    }

    tiny_array_impl & operator=(value_type const (&v)[static_size])
    {
        init_impl(v);
        return *this;
    }

    template <class U, bool OTHER_OWNS_MEMORY>
    tiny_array_impl & operator=(tiny_array_impl<U, OTHER_OWNS_MEMORY, N...> const & other)
    {
        xtensor_precondition(size() == other.size(),
            "tiny_array_impl::operator=(): size mismatch.");
        init_impl(other.begin());
        return *this;
    }

        /** Reset to the other pointer.
        */
    template <bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< !owns_memory >>
    void reset(pointer other)
    {
        data_ = other;
    }

    template <class OTHER, bool OTHER_OWNS_MEMORY, int ... M>
    constexpr bool
    is_same_shape(tiny_array_impl<OTHER, OTHER_OWNS_MEMORY, M...> const &) const
    {
        return false;
    }

    template <class OTHER, bool OTHER_OWNS_MEMORY>
    constexpr bool
    is_same_shape(tiny_array_impl<OTHER, OTHER_OWNS_MEMORY, N...> const &) const
    {
        return true;
    }

    template <class OTHER, bool OTHER_OWNS_MEMORY>
    bool
    is_same_shape(tiny_array_impl<OTHER, OTHER_OWNS_MEMORY, runtime_size> const & other) const
    {
        return sizeof...(N) == 1 && size() == other.size();
    }

    tiny_array_impl & init(value_type v = value_type())
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = v;
        return *this;
    }

    template <class ... V>
    tiny_array_impl & init(value_type v0, value_type v1, V... v)
    {
        static_assert(sizeof...(V)+2 == static_size,
                      "tiny_array_impl::init(): wrong number of arguments.");
        init_impl<0>(v0, v1, v...);
        return *this;
    }

    template <class Iterator>
    tiny_array_impl & init(Iterator first, Iterator end)
    {
        const index_t range = std::distance(first, end);
        if(range == 1)
        {
            init(static_cast<value_type>(*first));
        }
        else
        {
            xtensor_precondition(range == static_size,
                "tiny_array_impl::init(): size mismatch.");
            init_impl(first);
        }
        return *this;
    }

    // index access

    reference operator[](index_t i)
    {
        return data_[i];
    }

    constexpr const_reference operator[](index_t i) const
    {
        return data_[i];
    }

    reference at(index_t i)
    {
        if(i < 0 || i >= static_size)
            throw std::out_of_range("tiny_array_impl::at()");
        return data_[i];
    }

    const_reference at(index_t i) const
    {
        if(i < 0 || i >= static_size)
            throw std::out_of_range("tiny_array_impl::at()");
        return data_[i];
    }

    reference operator[](index_t const (&i)[static_ndim])
    {
        return data_[shape_helper::offset(i)];
    }

    constexpr const_reference operator[](index_t const (&i)[static_ndim]) const
    {
        return data_[shape_helper::offset(i)];
    }

    reference at(index_t const (&i)[static_ndim])
    {
        return at(shape_helper::offset(i));
    }

    const_reference at(index_t const (&i)[static_ndim]) const
    {
        return at(shape_helper::offset(i));
    }

    reference operator[](index_type const & i)
    {
        return data_[shape_helper::offset(i.data())];
    }

    constexpr const_reference operator[](index_type const & i) const
    {
        return data_[shape_helper::offset(i.data())];
    }

    reference at(index_type const & i)
    {
        return at(shape_helper::offset(i.data()));
    }

    const_reference at(index_type const & i) const
    {
        return at(shape_helper::offset(i.data()));
    }

    template <class ... V>
    reference operator()(V...v)
    {
        static_assert(sizeof...(V) == static_ndim,
                      "tiny_array_impl::operator(): wrong number of arguments.");
        return data_[shape_helper::offset(v...)];
    }

    template <class ... V>
    constexpr const_reference operator()(V...v) const
    {
        static_assert(sizeof...(V) == static_ndim,
                      "tiny_array_impl::operator(): wrong number of arguments.");
        return data_[shape_helper::offset(v...)];
    }

        /** Get a view to the subarray with length <tt>(TO-FROM)</tt> starting at <tt>FROM</tt>.
            The bounds must fullfill <tt>0 <= FROM < TO <= static_size</tt>.
            Only available if <tt>static_ndim == 1</tt>.
        */
    template <int FROM, int TO>
    tiny_array_adaptor<value_type, TO-FROM>
    subarray() const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array_impl::subarray(): array must be 1-dimensional.");
        static_assert(FROM >= 0 && FROM < TO && TO <= static_size,
            "tiny_array_impl::subarray(): range out of bounds.");
        return tiny_array_adaptor<value_type, TO-FROM>(const_cast<VALUETYPE*>(data_)+FROM);
    }

    tiny_array_adaptor<value_type, runtime_size>
    subarray(index_t FROM, index_t TO) const
    {
        xtensor_precondition(FROM >= 0 && FROM < TO && TO <= static_size,
                      "tiny_array_impl::subarray(): range out of bounds.");
        return tiny_array_adaptor<value_type, runtime_size>(TO-FROM, const_cast<VALUETYPE*>(data_)+FROM);
    }

    template<int M = static_ndim>
    tiny_array<value_type, static_size-1>
    erase(index_t m) const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array_impl::erase(): array must be 1-dimensional.");
        xtensor_precondition(m >= 0 && m < static_size, "tiny_array::erase(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+").");
        tiny_array<value_type, static_size-1> res(static_size-1, dont_init);
        for(int k=0; k<m; ++k)
            res[k] = data_[k];
        for(index_t k=m; k<static_size-1; ++k)
            res[k] = data_[k+1];
        return res;
    }

    template<int M = static_ndim>
    tiny_array<value_type, static_size-1>
    pop_front() const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array_impl::pop_front(): array must be 1-dimensional.");
        return erase(0);
    }

    template<int M = static_ndim>
    tiny_array<value_type, static_size-1>
    pop_back() const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array_impl::pop_back(): array must be 1-dimensional.");
        return erase(size()-1);
    }

    template<int M = static_ndim>
    tiny_array<value_type, static_size+1>
    insert(index_t m, value_type v) const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array_impl::insert(): array must be 1-dimensional.");
        xtensor_precondition(m >= 0 && m <= static_size, "tiny_array::insert(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+"].");
        tiny_array<value_type, static_size+1> res(dont_init);
        for(int k=0; k<m; ++k)
            res[k] = data_[k];
        res[m] = v;
        for(index_t k=m; k<static_size; ++k)
            res[k+1] = data_[k];
        return res;
    }

    template <class V, bool OM, int M>
    inline
    tiny_array<value_type, static_size>
    transpose(tiny_array_impl<V, OM, M> const & permutation) const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array::transpose(): only allowed for 1-dimensional arrays.");
        static_assert(M == static_size || M == runtime_size,
            "tiny_array::transpose(): size mismatch.");
        XTENSOR_ASSERT_RUNTIME_SIZE(M, size() == 0 || size() == permutation.size(),
            "tiny_array::transpose(): size mismatch.");
        tiny_array<value_type, static_size> res(dont_init);
        for(int k=0; k < size(); ++k)
        {
            XTENSOR_ASSERT_MSG(permutation[k] >= 0 && permutation[k] < size(),
                "transpose():  Permutation index out of bounds");
            res[k] = (*this)[permutation[k]];
        }
        return res;
    }

    // boiler plate

    iterator begin() { return data_; }
    iterator end()   { return data_ + static_size; }
    const_iterator begin() const { return data_; }
    const_iterator end()   const { return data_ + static_size; }
    const_iterator cbegin() const { return data_; }
    const_iterator cend()   const { return data_ + static_size; }

    reverse_iterator rbegin() { return reverse_iterator(data_ + static_size); }
    reverse_iterator rend()   { return reverse_iterator(data_); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(data_ + static_size); }
    const_reverse_iterator rend()   const { return const_reverse_iterator(data_); }
    const_reverse_iterator crbegin() const { return const_reverse_iterator(data_ + static_size); }
    const_reverse_iterator crend()   const { return const_reverse_iterator(data_); }

    pointer data() { return data_; }
    const_pointer data() const { return data_; }

    reference front() { return data_[0]; }
    reference back()  { return data_[static_size-1]; }
    constexpr const_reference front() const { return data_[0]; }
    constexpr const_reference back()  const { return data_[static_size-1]; }

    constexpr bool       empty() const { return static_size == 0; }
    constexpr index_t size()  const { return static_size; }
    constexpr index_t max_size()  const { return static_size; }
    constexpr index_type shape() const { return index_type{ N... }; }
    constexpr index_t ndim()  const { return static_ndim; }

    tiny_array_impl & reverse()
    {
        using std::swap;
        index_t i=0, j=size()-1;
        while(i < j)
             swap(data_[i++], data_[j--]);
        return *this;
    }

    void swap(tiny_array_impl & other)
    {
        using std::swap;
        for(int k=0; k<static_size; ++k)
        {
            swap(data_[k], other[k]);
        }
    }

    template <class OTHER, bool OTHER_OWNS_MEMORY>
    void swap(tiny_array_impl<OTHER, OTHER_OWNS_MEMORY, N...> & other)
    {
        for(int k=0; k<static_size; ++k)
        {
            promote_t<value_type, OTHER> t = data_[k];
            data_[k] = static_cast<value_type>(other[k]);
            other[k] = static_cast<OTHER>(t);
        }
    }

        /// factory function for fixed-size unit matrix
    template <int SIZE>
    static inline
    tiny_array<value_type, SIZE, SIZE>
    eye()
    {
        tiny_array<value_type, SIZE, SIZE> res;
        for(int k=0; k<SIZE; ++k)
            res(k,k) = 1;
        return res;
    }

        /// factory function for the fixed-size k-th unit vector
    template <int SIZE=static_size>
    static inline
    tiny_array<value_type, SIZE>
    unit_vector(index_t k)
    {
        tiny_array<value_type, SIZE> res;
        res(k) = 1;
        return res;
    }

        /// factory function for the k-th unit vector
        // (for compatibility with tiny_array<..., runtime_size>)
    static inline
    tiny_array<value_type, static_size>
    unit_vector(tags::size_proxy const & size, index_t k)
    {
        XTENSOR_ASSERT_MSG(size.value == static_size,
            "tiny_array::unit_vector(): size mismatch.");
        tiny_array<value_type, static_size> res;
        res(k) = 1;
        return res;
    }

        /// factory function for fixed-size linear sequence starting at <tt>start</tt> with stepsize <tt>step</tt>
    static inline
    tiny_array_impl<value_type, true, N...>
    linear_sequence(value_type start = value_type(), value_type step = value_type(1))
    {
        tiny_array_impl<value_type, true, N...> res(dont_init);
        for(index_t k=0; k < static_size; ++k, start += step)
            res[k] = start;
        return res;
    }

        /// factory function for fixed-size linear sequence ending at <tt>end-1</tt>
    static inline
    tiny_array_impl<value_type, true, N...>
    range(value_type end)
    {
        value_type start = end - static_cast<value_type>(static_size);
        tiny_array_impl<value_type, true, N...> res(dont_init);
        for(index_t k=0; k < static_size; ++k, ++start)
            res[k] = start;
        return res;
    }

  protected:
    data_array_type data_;
};

/********************************************************/
/*                                                      */
/*                tiny_array_impl output                */
/*                                                      */
/********************************************************/

template <class T, bool OWNS_MEMORY, int ... N>
std::ostream & operator<<(std::ostream & o, tiny_array_impl<T, OWNS_MEMORY, N...> const & v)
{
    o << "{";
    if(v.size() > 0)
        o << promote_t<T>(v[0]);
    for(int i=1; i < v.size(); ++i)
        o << ", " << promote_t<T>(v[i]);
    o << "}";
    return o;
}

template <class T, bool OWNS_MEMORY, int N1, int N2>
std::ostream & operator<<(std::ostream & o, tiny_array_impl<T, OWNS_MEMORY, N1, N2> const & v)
{
    o << "{";
    for(int i=0; N2>0 && i<N1; ++i)
    {
        if(i > 0)
            o << ",\n ";
        o << promote_t<T>(v(i,0));
        for(int j=1; j<N2; ++j)
        {
            o << ", " << promote_t<T>(v(i, j));
        }
    }
    o << "}";
    return o;
}

/********************************************************/
/*                                                      */
/*         tiny_array_impl<..., runtime_size>           */
/*                                                      */
/********************************************************/

/** \brief Specialization of tiny_array for dynamic arrays.
    \ingroup RangesAndPoints

    This class is typically used via the type alias \ref tiny_array or \ref tiny_matrix.

    This class encapsulates an dynamicaly allocazed array of the specified
    VALUETYPE whose size is specified at runtime.
    The interface conforms to STL vector, except that there are no functions
    that change the size of a tiny_array_impl.

    \ref TinyArrayOperators "Arithmetic operations"
    on TinyArrays are defined as component-wise applications of these
    operations.

    See also:<br>
    <UL style="list-style-image:url(documents/bullet.gif)">
        <LI> \ref xt::tiny_array
        <LI> \ref xt::tiny_array_adaptor
        <LI> \ref TinyArrayOperators
    </UL>
**/
template <class VALUETYPE, bool OWNS_MEMORY>
class tiny_array_impl<VALUETYPE, OWNS_MEMORY, runtime_size>
: public tiny_array_tag
{
  public:

    template <class NEW_VALUETYPE>
    using as_type = tiny_array_impl<NEW_VALUETYPE, OWNS_MEMORY, runtime_size>;

    using value_type             = VALUETYPE;
    using const_value_type       = typename std::add_const<VALUETYPE>::type;
    using reference              = value_type &;
    using const_reference        = const_value_type &;
    using pointer                = value_type *;
    using const_pointer          = const_value_type *;
    using iterator               = value_type *;
    using const_iterator         = const_value_type *;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using size_type              = std::size_t;
    using difference_type        = std::ptrdiff_t;
    using index_type             = index_t;

    static const index_t static_size  = runtime_size;
    static const index_t static_ndim  = 1;
    static const bool may_use_uninitialized_memory =
                                   detail::may_use_uninitialized_memory<VALUETYPE>::value;

  protected:

    template <int LEVEL, class ... V2>
    void init_impl(VALUETYPE v1, V2... v2)
    {
        data_[LEVEL] = v1;
        init_impl<LEVEL+1>(v2...);
    }

    template <int LEVEL>
    void init_impl(VALUETYPE v1)
    {
        data_[LEVEL] = v1;
    }

    template <class U,
              XTENSOR_REQUIRE<iterator_concept<U>::value>>
    void init_impl(U u)
    {
        for(index_t k=0; k < size_; ++k, ++u)
            data_[k] = static_cast<VALUETYPE>(*u);
    }

  public:

    // constructors

    tiny_array_impl()
    : size_(0)
    , data_(nullptr)
    {}

    tiny_array_impl(tiny_array_impl && rhs)
    : tiny_array_impl()
    {
        rhs.swap(*this);
    }

    tiny_array_impl(tiny_array_impl const & rhs )
    : size_(rhs.size())
    , data_(OWNS_MEMORY ? alloc_.allocate(size_) : const_cast<pointer>(rhs.data()))
    {
        if(OWNS_MEMORY)
            std::uninitialized_copy(rhs.begin(), rhs.end(), begin());
    }

    // constructors when array does not own memory

    template <bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< !owns_memory >>
    tiny_array_impl(index_t size, pointer data)
    : size_(size)
    , data_(data)
    {}

    // constructors when array owns memory

    template <bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< owns_memory >>
    explicit
    tiny_array_impl(index_t size,
                    value_type const & initial = value_type())
    : size_(size)
    , data_(alloc_.allocate(size))
    {
        std::uninitialized_fill(begin(), end(), initial);
    }

    template <bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< owns_memory >>
    explicit
    tiny_array_impl(tags::size_proxy const & size,
                    value_type const & initial = value_type())
    : tiny_array_impl(size.value, initial)
    {}

    template <bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< owns_memory >>
    tiny_array_impl(index_t size, skip_initialization_tag)
    : size_(size)
    , data_(alloc_.allocate(size))
    {
        if(!may_use_uninitialized_memory)
            std::uninitialized_fill(begin(), end(), value_type());
    }

    template <class U, bool OTHER_OWNS_MEMORY, int ... N,
              bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< owns_memory >>
    tiny_array_impl(tiny_array_impl<U, OTHER_OWNS_MEMORY, N...> const & other)
    : tiny_array_impl(other.begin(), other.end())
    {}

    template <class U,
              bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE<iterator_concept<U>::value && owns_memory> >
    tiny_array_impl(U begin, U end)
    : size_(std::distance(begin, end))
    , data_(alloc_.allocate(size_))
    {
        for(int i=0; i<size_; ++i, ++begin)
            new(data_+i) value_type(static_cast<value_type>(*begin));
    }

    template <class U,
              bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE<iterator_concept<U>::value && owns_memory> >
    tiny_array_impl(U begin, U end, reverse_copy_tag)
    : size_(std::distance(begin, end))
    , data_(alloc_.allocate(size_))
    {
        for(int i=0; i<size_; ++i)
        {
            --end;
            new(data_+i) value_type(static_cast<value_type>(*end));
        }
    }

    template <class U, size_t SIZE,
              bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< owns_memory >>
    tiny_array_impl(const U (&u)[SIZE])
    : tiny_array_impl(u, u+SIZE)
    {}

    template <class U,
              bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< owns_memory >>
    tiny_array_impl(std::initializer_list<U> rhs)
    : tiny_array_impl(rhs.begin(), rhs.end())
    {}

    // assignment
    tiny_array_impl & operator=(value_type const & v)
    {
        init(v);
        return *this;
    }

    tiny_array_impl & operator=(tiny_array_impl const & rhs)
    {
        if(this == &rhs)
            return *this;
        if(size_ != rhs.size())
            tiny_array_impl(rhs).swap(*this);
        else
            init(rhs.begin(), rhs.end());
        return *this;
    }

    tiny_array_impl & operator=(tiny_array_impl && rhs)
    {
        if(size_ != rhs.size())
            rhs.swap(*this);
        else
            init(rhs.begin(), rhs.end());
        return *this;
    }

    template <class U, bool OTHER_OWNS_MEMORY, int ... N,
              bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< owns_memory >>
    tiny_array_impl & operator=(tiny_array_impl<U, OTHER_OWNS_MEMORY, N...> const & rhs)
    {
        if(size_!= rhs.size())
            tiny_array_impl(rhs).swap(*this);
        else
            init(rhs.begin(), rhs.end());
        return *this;
    }

    ~tiny_array_impl()
    {
        if(!OWNS_MEMORY)
            return;
        if(!may_use_uninitialized_memory)
        {
            for(index_t i=0; i<size_; ++i)
                (data_+i)->~value_type();
        }
        alloc_.deallocate(data_, size_);
    }

    template <bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< owns_memory >>
    void resize(size_t new_size)
    {
        if(new_size != size())
        {
            tiny_array_impl(new_size).swap(*this);
        }
    }

        /** Reset pointer.
        */
    template <bool owns_memory=OWNS_MEMORY,
              XTENSOR_REQUIRE< !owns_memory >>
    void reset(index_t size, pointer p)
    {
        size_ = size;
        data_ = p;
    }

    template <class OTHER, bool OTHER_OWNS_MEMORY, int ... M>
    bool
    is_same_shape(tiny_array_impl<OTHER, OTHER_OWNS_MEMORY, M...> const & other) const
    {
        return sizeof...(M) == 1 && size() == other.size();
    }

    template <class OTHER, bool OTHER_OWNS_MEMORY>
    bool
    is_same_shape(tiny_array_impl<OTHER, OTHER_OWNS_MEMORY, runtime_size> const & other) const
    {
        return size() == other.size();
    }

    tiny_array_impl & init(value_type v = value_type())
    {
        for(int i=0; i<size_; ++i)
            data_[i] = v;
        return *this;
    }

    template <class ... V>
    tiny_array_impl & init(value_type v0, value_type v1, V... v)
    {
        xtensor_precondition(sizeof...(V)+2 == size_,
                      "tiny_array_impl::init(): wrong number of arguments.");
        init_impl<0>(v0, v1, v...);
        return *this;
    }

    template <class ITERATOR,
              XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value>>
    tiny_array_impl & init(ITERATOR first, ITERATOR end)
    {
        index_t range = std::distance(first, end);
        if(range == 1)
            init(static_cast<value_type>(*first));
        else if(range == size_)
            init_impl(first);
        else
            xtensor_precondition(false,
                "tiny_array_impl::init(): size mismatch.");
        return *this;
    }

    template <class V>
    tiny_array_impl & init(std::initializer_list<V> l)
    {
        return init(l.begin(), l.end());
    }

    // index access

    reference operator[](index_t i)
    {
        return data_[i];
    }

    constexpr const_reference operator[](index_t i) const
    {
        return data_[i];
    }

    reference at(index_t i)
    {
        if(i < 0 || i >= size_)
            throw std::out_of_range("tiny_array_impl::at()");
        return data_[i];
    }

    const_reference at(index_t i) const
    {
        if(i < 0 || i >= size_)
            throw std::out_of_range("tiny_array_impl::at()");
        return data_[i];
    }

        /** Get a view to the subarray with length <tt>(TO-FROM)</tt> starting at <tt>FROM</tt>.
            The bounds must fullfill <tt>0 <= FROM < TO <= size_</tt>.
        */
    template <int FROM, int TO>
    tiny_array_adaptor<value_type, TO-FROM>
    subarray() const
    {
        static_assert(FROM >= 0 && FROM < TO,
            "tiny_array_impl::subarray(): range out of bounds.");
        xtensor_precondition(TO <= size_,
            "tiny_array_impl::subarray(): range out of bounds.");
        return tiny_array_adaptor<value_type, TO-FROM>(data_+FROM);
    }

        /** Get a view to the subarray with length <tt>(TO-FROM)</tt> starting at <tt>FROM</tt>.
            The bounds must fullfill <tt>0 <= FROM < TO <= size_</tt>.
        */
    tiny_array_adaptor<value_type, runtime_size>
    subarray(index_t FROM, index_t TO) const
    {
        xtensor_precondition(FROM >= 0 && FROM < TO && TO <= size_,
            "tiny_array_impl::subarray(): range out of bounds.");
        return tiny_array_adaptor<value_type, runtime_size>(TO-FROM, const_cast<pointer>(data_)+FROM);
    }


    tiny_array<value_type, runtime_size>
    erase(index_t m) const
    {
        xtensor_precondition(m >= 0 && m < size(), "tiny_array::erase(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+").");
        tiny_array<value_type, runtime_size> res(size()-1, dont_init);
        for(int k=0; k<m; ++k)
            res[k] = data_[k];
        for(index_t k=m+1; k<size(); ++k)
            res[k-1] = data_[k];
        return res;
    }

    tiny_array<value_type, runtime_size>
    pop_front() const
    {
        return erase(0);
    }

    tiny_array<value_type, runtime_size>
    pop_back() const
    {
        return erase(size()-1);
    }

    tiny_array<value_type, runtime_size>
    insert(index_t m, value_type v) const
    {
        xtensor_precondition(m >= 0 && m <= size(), "tiny_array::insert(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+"].");
        tiny_array<value_type, runtime_size> res(size()+1, dont_init);
        for(int k=0; k<m; ++k)
            res[k] = data_[k];
        res[m] = v;
        for(index_t k=m; k<size(); ++k)
            res[k+1] = data_[k];
        return res;
    }

    template <class V, bool OM, int M>
    inline
    tiny_array<value_type, runtime_size>
    transpose(tiny_array_impl<V, OM, M> const & permutation) const
    {
        xtensor_precondition(size() == 0 || size() == permutation.size(),
            "tiny_array::transpose(): size mismatch.");
        tiny_array<value_type, runtime_size> res(size(), dont_init);
        for(index_t k=0; k < size(); ++k)
        {
            XTENSOR_ASSERT_MSG(permutation[k] >= 0 && permutation[k] < size(),
                "transpose():  Permutation index out of bounds");
            res[k] = (*this)[permutation[k]];
        }
        return res;
    }

    // boiler plate

    iterator begin() { return data_; }
    iterator end()   { return data_ + size_; }
    const_iterator begin() const { return data_; }
    const_iterator end()   const { return data_ + size_; }
    const_iterator cbegin() const { return data_; }
    const_iterator cend()   const { return data_ + size_; }

    reverse_iterator rbegin() { return reverse_iterator(data_ + size_); }
    reverse_iterator rend()   { return reverse_iterator(data_); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(data_ + size_); }
    const_reverse_iterator rend()   const { return const_reverse_iterator(data_); }
    const_reverse_iterator crbegin() const { return const_reverse_iterator(data_ + size_); }
    const_reverse_iterator crend()   const { return const_reverse_iterator(data_); }

    pointer data() { return data_; }
    const_pointer data() const { return data_; }

    reference front() { return data_[0]; }
    reference back()  { return data_[size_-1]; }
    const_reference front() const { return data_[0]; }
    const_reference back()  const { return data_[size_-1]; }

    bool       empty() const { return size_ == 0; }
    index_t size()  const { return size_; }
    index_t max_size()  const { return size_; }
    index_t ndim()  const { return static_ndim; }

    tiny_array_impl & reverse()
    {
        using std::swap;
        index_t i=0, j=size_-1;
        while(i < j)
             swap(data_[i++], data_[j--]);
        return *this;
    }

    void swap(tiny_array_impl & other)
    {
        using std::swap;
        swap(size_, other.size_);
        swap(data_, other.data_);
    }

        /// factory function for the fixed-size k-th unit vector
    static inline
    tiny_array<value_type, runtime_size>
    unit_vector(tags::size_proxy const & size, index_t k)
    {
        tiny_array<value_type, runtime_size> res(size.value);
        res[k] = 1;
        return res;
    }

        /// factory function for a linear sequence from <tt>begin</tt> to <tt>end</tt>
        /// (exclusive) with stepsize <tt>step</tt>
    static inline
    tiny_array<value_type, runtime_size>
    range(value_type begin,
          value_type end,
          value_type step = value_type(1))
    {
        using namespace cmath;
        xtensor_precondition(step != 0,
            "tiny_array::range(): step must be non-zero.");
        xtensor_precondition((step > 0 && begin <= end) || (step < 0 && begin >= end),
            "tiny_array::range(): sign mismatch between step and (end-begin).");
        index_t size = floor((abs(end-begin+step)-1)/abs(step));
        tiny_array<value_type, runtime_size> res(size, dont_init);
        for(index_t k=0; k < size; ++k, begin += step)
            res[k] = begin;
        return res;
    }

        /// factory function for a linear sequence from 0 to <tt>end</tt>
        /// (exclusive) with stepsize 1
    static inline
    tiny_array<value_type, runtime_size>
    range(value_type end)
    {
        xtensor_precondition(end >= 0,
            "tiny_array::range(): end must be non-negative.");
        tiny_array<value_type, runtime_size> res(end, dont_init);
        auto begin = value_type();
        for(index_t k=0; k < res.size(); ++k, ++begin)
            res[k] = begin;
        return res;
    }

  protected:
    // FIXME: implement an optimized allocator
    // FIXME: (look at Alexandrescu's Loki library or Kolmogorov's code)
    std::allocator<value_type> alloc_;
    index_t size_;
    pointer data_;
};

/********************************************************/
/*                                                      */
/*                tiny_array Comparison                 */
/*                                                      */
/********************************************************/

/** \addtogroup TinyArrayOperators Functions for tiny_array

    \brief Implement basic arithmetic and equality for tiny_array.

    These functions fulfill the requirements of a Linear Space (vector space).
    Return types are determined according to \ref promote_t or \ref real_promote_t.
*/
//@{

    /// element-wise equal
template <class V1, bool OM1, class V2, bool OM2, int ...M, int ... N>
inline bool
operator==(tiny_array_impl<V1, OM1, M...> const & l,
           tiny_array_impl<V2, OM2, N...> const & r)
{
    if(!l.is_same_shape(r))
        return false;
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r[k])
            return false;
    return true;
}

    /// element-wise equal to a constant
template <class V1, bool OM1, class V2, int ...M,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
operator==(tiny_array_impl<V1, OM1, M...> const & l,
           V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r)
            return false;
    return true;
}

    /// element-wise equal to a constant
template <class V1, class V2, bool OM2, int ...M,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
operator==(V1 const & l,
           tiny_array_impl<V2, OM2, M...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if(l != r[k])
            return false;
    return true;
}

    /// element-wise not equal
template <class V1, bool OM1, class V2, bool OM2, int ... M, int ... N>
inline bool
operator!=(tiny_array_impl<V1, OM1, M...> const & l,
           tiny_array_impl<V2, OM2, N...> const & r)
{
    if(!l.is_same_shape(r))
        return true;
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r[k])
            return true;
    return false;
}

    /// element-wise not equal to a constant
template <class V1, bool OM1, class V2, int ... M,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
operator!=(tiny_array_impl<V1, OM1, M...> const & l,
           V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r)
            return true;
    return false;
}

    /// element-wise not equal to a constant
template <class V1, class V2, bool OM2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
operator!=(V1 const & l,
           tiny_array_impl<V2, OM2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if(l != r[k])
            return true;
    return false;
}

    /// lexicographical less
template <class V1, bool OM1, class V2, bool OM2, int ... N>
inline bool
operator<(tiny_array_impl<V1, OM1, N...> const & l,
          tiny_array_impl<V2, OM2, N...> const & r)
{
    const index_t min_size = min(l.size(), r.size());
    for(int k = 0; k < min_size; ++k)
    {
        if(l[k] < r[k])
            return true;
        if(r[k] < l[k])
            return false;
    }
    return (l.size() < r.size());
}

    /// lexicographical less-equal
template <class V1, bool OM1, class V2, bool OM2, int ... N>
inline bool
operator<=(tiny_array_impl<V1, OM1, N...> const & l,
           tiny_array_impl<V2, OM2, N...> const & r)
{
    return !(r < l);
}

    /// lexicographical greater
template <class V1, bool OM1, class V2, bool OM2, int ... N>
inline bool
operator>(tiny_array_impl<V1, OM1, N...> const & l,
          tiny_array_impl<V2, OM2, N...> const & r)
{
    return r < l;
}

    /// lexicographical greater-equal
template <class V1, bool OM1, class V2, bool OM2, int ... N>
inline bool
operator>=(tiny_array_impl<V1, OM1, N...> const & l,
           tiny_array_impl<V2, OM2, N...> const & r)
{
    return !(l < r);
}

    /// check if all elements are non-zero (or 'true' if V is bool)
template <class V, bool OM, int ... N>
inline bool
all(tiny_array_impl<V, OM, N...> const & t)
{
    for(int i=0; i<t.size(); ++i)
        if(t[i] == V())
            return false;
    return true;
}

    /// check if at least one element is non-zero (or 'true' if V is bool)
template <class V, bool OM, int ... N>
inline bool
any(tiny_array_impl<V, OM, N...> const & t)
{
    for(int i=0; i<t.size(); ++i)
        if(t[i] != V())
            return true;
    return false;
}

    /// check if all elements are zero (or 'false' if V is bool)
template <class V, bool OM, int ... N>
inline bool
all_zero(tiny_array_impl<V, OM, N...> const & t)
{
    for(int i=0; i<t.size(); ++i)
        if(t[i] != V())
            return false;
    return true;
}

    /// pointwise less-than
template <class V1, bool OM1, class V2, bool OM2, int ... N>
inline bool
all_less(tiny_array_impl<V1, OM1, N...> const & l,
         tiny_array_impl<V2, OM2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "tiny_array_base::all_less(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
        if (l[k] >= r[k])
            return false;
    return true;
}

    /// pointwise less than a constant
    /// (typically used to check negativity with `r = 0`)
template <class V1, bool OM1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_less(tiny_array_impl<V1, OM1, N...> const & l,
         V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] >= r)
            return false;
    return true;
}

    /// constant pointwise less than the vector
    /// (typically used to check positivity with `l = 0`)
template <class V1, class V2, bool OM2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_less(V1 const & l,
         tiny_array_impl<V2, OM2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l >= r[k])
            return false;
    return true;
}

    /// pointwise less-equal
template <class V1, bool OM1, class V2, bool OM2, int ... N>
inline bool
all_less_equal(tiny_array_impl<V1, OM1, N...> const & l,
               tiny_array_impl<V2, OM2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "tiny_array_base::all_less_equal(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
        if (l[k] > r[k])
            return false;
    return true;
}

    /// pointwise less-equal with a constant
    /// (typically used to check non-positivity with `r = 0`)
template <class V1, bool OM1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_less_equal(tiny_array_impl<V1, OM1, N...> const & l,
               V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] > r)
            return false;
    return true;
}

    /// pointwise less-equal with a constant
    /// (typically used to check non-negativity with `l = 0`)
template <class V1, class V2, bool OM2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_less_equal(V1 const & l,
               tiny_array_impl<V2, OM2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l > r[k])
            return false;
    return true;
}

    /// pointwise greater-than
template <class V1, bool OM1, class V2, bool OM2, int ... N>
inline bool
all_greater(tiny_array_impl<V1, OM1, N...> const & l,
            tiny_array_impl<V2, OM2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "tiny_array_base::all_greater(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
        if (l[k] <= r[k])
            return false;
    return true;
}

    /// pointwise greater-than with a constant
    /// (typically used to check positivity with `r = 0`)
template <class V1, bool OM1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_greater(tiny_array_impl<V1, OM1, N...> const & l,
            V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] <= r)
            return false;
    return true;
}

    /// constant pointwise greater-than a vector
    /// (typically used to check negativity with `l = 0`)
template <class V1, class V2, bool OM2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_greater(V1 const & l,
            tiny_array_impl<V2, OM2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l <= r[k])
            return false;
    return true;
}

    /// pointwise greater-equal
template <class V1, bool OM1, class V2, bool OM2, int ... N>
inline bool
all_greater_equal(tiny_array_impl<V1, OM1, N...> const & l,
                  tiny_array_impl<V2, OM2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "tiny_array_base::all_greater_equal(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
        if (l[k] < r[k])
            return false;
    return true;
}

    /// pointwise greater-equal with a constant
    /// (typically used to check non-negativity with `r = 0`)
template <class V1, bool OM1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_greater_equal(tiny_array_impl<V1, OM1, N...> const & l,
                  V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] < r)
            return false;
    return true;
}

    /// pointwise greater-equal with a constant
    /// (typically used to check non-positivity with `l = 0`)
template <class V1, class V2, bool OM2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_greater_equal(V1 const & l,
                  tiny_array_impl<V2, OM2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l < r[k])
            return false;
    return true;
}

template <class V1, bool OM1, class V2, bool OM2, int ... N>
inline bool
isclose(tiny_array_impl<V1, OM1, N...> const & l,
        tiny_array_impl<V2, OM2, N...> const & r,
        promote_t<V1, V2> epsilon = 2.0*std::numeric_limits<promote_t<V1, V2> >::epsilon())
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "tiny_array_base::isclose(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
        if(!isclose(l[k], r[k], epsilon, epsilon))
            return false;
    return true;
}

/********************************************************/
/*                                                      */
/*                 tiny_array-Arithmetic                */
/*                                                      */
/********************************************************/

/** \addtogroup TinyArrayOperators
 */
//@{

#ifdef DOXYGEN
// Declare arithmetic functions for documentation,
// the implementations are provided by a macro below.

    /// scalar add-assignment
template <class V1, bool OWNS_MEMORY, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
tiny_array_impl<V1, OWNS_MEMORY, N...> &
operator+=(tiny_array_impl<V1, OWNS_MEMORY, N...> & l,
           V2 r);

    /// element-wise add-assignment
template <class V1, bool OWNS_MEMORY, class V2, bool OTHER_OWNS_MEMORY, int ... N>
tiny_array_impl<V1, OWNS_MEMORY, N...> &
operator+=(tiny_array_impl<V1, OWNS_MEMORY, N...> & l,
           tiny_array_impl<V2, OTHER_OWNS_MEMORY, N...> const & r);

    /// element-wise addition
template <class V1, bool OM1, class V2, bool OM2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator+(tiny_array_impl<V1, OM1, N...> const & l,
          tiny_array_impl<V2, OM2, N...> const & r);

    /// element-wise scalar addition
template <class V1, bool OM1, class V2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator+(tiny_array_impl<V1, OM1, N...> const & l,
          V2 r);

    /// element-wise left scalar addition
template <class V1, class V2, bool OM2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator+(V1 l,
          tiny_array_impl<V2, OM2, N...> const & r);

    /// scalar subtract-assignment
template <class V1, bool OWNS_MEMORY, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
tiny_array_impl<V1, OWNS_MEMORY, N...> &
operator-=(tiny_array_impl<V1, OWNS_MEMORY, N...> & l,
           V2 r);

    /// element-wise subtract-assignment
template <class V1, bool OWNS_MEMORY, class V2, bool OTHER_OWNS_MEMORY, int ... N>
tiny_array_impl<V1, OWNS_MEMORY, N...> &
operator-=(tiny_array_impl<V1, OWNS_MEMORY, N...> & l,
           tiny_array_impl<V2, OTHER_OWNS_MEMORY, N...> const & r);

    /// element-wise subtraction
template <class V1, bool OM1, class V2, bool OM2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator-(tiny_array_impl<V1, OM1, N...> const & l,
          tiny_array_impl<V2, OM2, N...> const & r);

    /// element-wise scalar subtraction
template <class V1, bool OM1, class V2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator-(tiny_array_impl<V1, OM1, N...> const & l,
          V2 r);

    /// element-wise left scalar subtraction
template <class V1, class V2, bool OM2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator-(V1 l,
          tiny_array_impl<V2, OM2, N...> const & r);

    /// scalar multiply-assignment
template <class V1, bool OWNS_MEMORY, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
tiny_array_impl<V1, OWNS_MEMORY, N...> &
operator*=(tiny_array_impl<V1, OWNS_MEMORY, N...> & l,
           V2 r);

    /// element-wise multiply-assignment
template <class V1, bool OWNS_MEMORY, class V2, bool OTHER_OWNS_MEMORY, int ... N>
tiny_array_impl<V1, OWNS_MEMORY, N...> &
operator*=(tiny_array_impl<V1, OWNS_MEMORY, N...> & l,
           tiny_array_impl<V2, OTHER_OWNS_MEMORY, N...> const & r);

    /// element-wise multiplication
template <class V1, bool OM1, class V2, bool OM2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator*(tiny_array_impl<V1, OM1, N...> const & l,
          tiny_array_impl<V2, OM2, N...> const & r);

    /// element-wise scalar multiplication
template <class V1, bool OM1, class V2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator*(tiny_array_impl<V1, OM1, N...> const & l,
          V2 r);

    /// element-wise left scalar multiplication
template <class V1, class V2, bool OM2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator*(V1 l,
          tiny_array_impl<V2, OM2, N...> const & r);

    /// scalar divide-assignment
template <class V1, bool OWNS_MEMORY, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
tiny_array_impl<V1, OWNS_MEMORY, N...> &
operator/=(tiny_array_impl<V1, OWNS_MEMORY, N...> & l,
           V2 r);

    /// element-wise divide-assignment
template <class V1, bool OWNS_MEMORY, class V2, bool OTHER_OWNS_MEMORY, int ... N>
tiny_array_impl<V1, OWNS_MEMORY, N...> &
operator/=(tiny_array_impl<V1, OWNS_MEMORY, N...> & l,
           tiny_array_impl<V2, OTHER_OWNS_MEMORY, N...> const & r);

    /// element-wise division
template <class V1, bool OM1, class V2, bool OM2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator/(tiny_array_impl<V1, OM1, N...> const & l,
          tiny_array_impl<V2, OM2, N...> const & r);

    /// element-wise scalar division
template <class V1, bool OM1, class V2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator/(tiny_array_impl<V1, OM1, N...> const & l,
          V2 r);

    /// element-wise left scalar division
template <class V1, class V2, bool OM2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator/(V1 l,
          tiny_array_impl<V2, OM2, N...> const & r);

    /// scalar modulo-assignment
template <class V1, bool OWNS_MEMORY, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
tiny_array_impl<V1, OWNS_MEMORY, N...> &
operator%=(tiny_array_impl<V1, OWNS_MEMORY, N...> & l,
           V2 r);

    /// element-wise modulo-assignment
template <class V1, bool OWNS_MEMORY, class V2, bool OTHER_OWNS_MEMORY, int ... N>
tiny_array_impl<V1, OWNS_MEMORY, N...> &
operator%=(tiny_array_impl<V1, OWNS_MEMORY, N...> & l,
           tiny_array_impl<V2, OTHER_OWNS_MEMORY, N...> const & r);

    /// element-wise modulo
template <class V1, bool OM1, class V2, bool OM2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator%(tiny_array_impl<V1, OM1, N...> const & l,
          tiny_array_impl<V2, OM2, N...> const & r);

    /// element-wise scalar modulo
template <class V1, bool OM1, class V2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator%(tiny_array_impl<V1, OM1, N...> const & l,
          V2 r);

    /// element-wise left scalar modulo
template <class V1, class V2, bool OM2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator%(V1 l,
          tiny_array_impl<V2, OM2, N...> const & r);

#else

#define XTENSOR_TINYARRAY_OPERATORS(OP) \
template <class V1, bool OWNS_MEMORY, int ... N, class V2, \
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value && \
                          std::is_convertible<V2, V1>::value> > \
inline tiny_array_impl<V1, OWNS_MEMORY, N...> & \
operator OP##=(tiny_array_impl<V1, OWNS_MEMORY, N...> & l, \
               V2 r) \
{ \
    for(int i=0; i<l.size(); ++i) \
        l[i] OP##= r; \
    return l; \
} \
 \
template <class V1, bool OWNS_MEMORY, class V2, bool OTHER_OWNS_MEMORY, int ... N> \
inline tiny_array_impl<V1, OWNS_MEMORY, N...> &  \
operator OP##=(tiny_array_impl<V1, OWNS_MEMORY, N...> & l, \
               tiny_array_impl<V2, OTHER_OWNS_MEMORY, N...> const & r) \
{ \
    XTENSOR_ASSERT_MSG(l.size() == r.size(), \
        "tiny_array_impl::operator" #OP "=(): size mismatch."); \
    for(int i=0; i<l.size(); ++i) \
        l[i] OP##= r[i]; \
    return l; \
} \
template <class V1, bool OM1, class V2, bool OM2, int ... N> \
inline \
tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), true, N...> \
operator OP(tiny_array_impl<V1, OM1, N...> const & l, \
            tiny_array_impl<V2, OM2, N...> const & r) \
{ \
    tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), true, N...> res(l); \
    return res OP##= r; \
} \
 \
template <class V1, bool OM1, class V2, int ... N, \
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value> >\
inline \
tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), true, N...> \
operator OP(tiny_array_impl<V1, OM1, N...> const & l, \
            V2 r) \
{ \
    tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), true, N...> res(l); \
    return res OP##= r; \
} \
 \
template <class V1, class V2, bool OM2, int ... N, \
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value && \
                          !std::is_base_of<std::ios_base, V1>::value> >\
inline \
tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), true, N...> \
operator OP(V1 l, \
             tiny_array_impl<V2, OM2, N...> const & r) \
{ \
    tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), true, N...> res{l}; \
    return res OP##= r; \
} \
 \
template <class V1, class V2, bool OM2, \
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value && \
                          !std::is_base_of<std::ios_base, V1>::value> >\
inline \
tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), true, runtime_size> \
operator OP(V1 l, \
             tiny_array_impl<V2, OM2, runtime_size> const & r) \
{ \
    tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), true, runtime_size> res(tags::size=r.size(), l); \
    return res OP##= r; \
}

XTENSOR_TINYARRAY_OPERATORS(+)
XTENSOR_TINYARRAY_OPERATORS(-)
XTENSOR_TINYARRAY_OPERATORS(*)
XTENSOR_TINYARRAY_OPERATORS(/)
XTENSOR_TINYARRAY_OPERATORS(%)
XTENSOR_TINYARRAY_OPERATORS(&)
XTENSOR_TINYARRAY_OPERATORS(|)
XTENSOR_TINYARRAY_OPERATORS(^)
XTENSOR_TINYARRAY_OPERATORS(<<)
XTENSOR_TINYARRAY_OPERATORS(>>)

#undef XTENSOR_TINYARRAY_OPERATORS

#endif // DOXYGEN

    // define sqrt() explicitly because its return type
    // is needed for type inference
template <class V, bool OM, int ... N>
inline tiny_array_impl<real_promote_t<V>, true, N...>
sqrt(tiny_array_impl<V, OM, N...> const & v)
{
    using namespace cmath;
    tiny_array_impl<real_promote_t<V>, true, N...> res(v.size(), dont_init);
    for(int k=0; k < v.size(); ++k)
        res[k] = sqrt(v[k]);
    return res;
}

#define XTENSOR_TINYARRAY_UNARY_FUNCTION(FCT) \
template <class V, bool OM, int ... N> \
inline auto \
FCT(tiny_array_impl<V, OM, N...> const & v) \
{ \
    using namespace cmath; \
    tiny_array<bool_promote_t<decltype(FCT(*(V*)0))>, N...> res(v.size(), dont_init); \
    for(int k=0; k < v.size(); ++k) \
        res[k] = FCT(v[k]); \
    return res; \
}

XTENSOR_TINYARRAY_UNARY_FUNCTION(abs)
XTENSOR_TINYARRAY_UNARY_FUNCTION(fabs)

XTENSOR_TINYARRAY_UNARY_FUNCTION(cos)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sin)
XTENSOR_TINYARRAY_UNARY_FUNCTION(tan)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sin_pi)
XTENSOR_TINYARRAY_UNARY_FUNCTION(cos_pi)
XTENSOR_TINYARRAY_UNARY_FUNCTION(acos)
XTENSOR_TINYARRAY_UNARY_FUNCTION(asin)
XTENSOR_TINYARRAY_UNARY_FUNCTION(atan)

XTENSOR_TINYARRAY_UNARY_FUNCTION(cosh)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sinh)
XTENSOR_TINYARRAY_UNARY_FUNCTION(tanh)
XTENSOR_TINYARRAY_UNARY_FUNCTION(acosh)
XTENSOR_TINYARRAY_UNARY_FUNCTION(asinh)
XTENSOR_TINYARRAY_UNARY_FUNCTION(atanh)

XTENSOR_TINYARRAY_UNARY_FUNCTION(cbrt)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sq)
XTENSOR_TINYARRAY_UNARY_FUNCTION(elementwise_norm)
XTENSOR_TINYARRAY_UNARY_FUNCTION(elementwise_squared_norm)

XTENSOR_TINYARRAY_UNARY_FUNCTION(exp)
XTENSOR_TINYARRAY_UNARY_FUNCTION(exp2)
XTENSOR_TINYARRAY_UNARY_FUNCTION(expm1)
XTENSOR_TINYARRAY_UNARY_FUNCTION(log)
XTENSOR_TINYARRAY_UNARY_FUNCTION(log2)
XTENSOR_TINYARRAY_UNARY_FUNCTION(log10)
XTENSOR_TINYARRAY_UNARY_FUNCTION(log1p)
XTENSOR_TINYARRAY_UNARY_FUNCTION(logb)
XTENSOR_TINYARRAY_UNARY_FUNCTION(ilogb)

XTENSOR_TINYARRAY_UNARY_FUNCTION(ceil)
XTENSOR_TINYARRAY_UNARY_FUNCTION(floor)
XTENSOR_TINYARRAY_UNARY_FUNCTION(trunc)
XTENSOR_TINYARRAY_UNARY_FUNCTION(round)
XTENSOR_TINYARRAY_UNARY_FUNCTION(lround)
XTENSOR_TINYARRAY_UNARY_FUNCTION(llround)
XTENSOR_TINYARRAY_UNARY_FUNCTION(even)
XTENSOR_TINYARRAY_UNARY_FUNCTION(odd)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sign)
XTENSOR_TINYARRAY_UNARY_FUNCTION(signi)

XTENSOR_TINYARRAY_UNARY_FUNCTION(erf)
XTENSOR_TINYARRAY_UNARY_FUNCTION(erfc)
XTENSOR_TINYARRAY_UNARY_FUNCTION(tgamma)
XTENSOR_TINYARRAY_UNARY_FUNCTION(lgamma)

XTENSOR_TINYARRAY_UNARY_FUNCTION(conj)
XTENSOR_TINYARRAY_UNARY_FUNCTION(real)
XTENSOR_TINYARRAY_UNARY_FUNCTION(imag)
XTENSOR_TINYARRAY_UNARY_FUNCTION(arg)

#undef XTENSOR_TINYARRAY_UNARY_FUNCTION

    /// Arithmetic negation
template <class V, bool OM, int ... N>
inline
tiny_array_impl<V, true, N...>
operator-(tiny_array_impl<V, OM, N...> const & v)
{
    tiny_array_impl<V, true, N...> res(v.size(), dont_init);
    for(int k=0; k < v.size(); ++k)
        res[k] = -v[k];
    return res;
}

    /// Boolean negation
template <class V, bool OM, int ... N>
inline
tiny_array_impl<V, true, N...>
operator!(tiny_array_impl<V, OM, N...> const & v)
{
    tiny_array_impl<V, true, N...> res(v.size(), dont_init);
    for(int k=0; k < v.size(); ++k)
        res[k] = !v[k];
    return res;
}

    /// Bitwise negation
template <class V, bool OM, int ... N>
inline
tiny_array_impl<V, true, N...>
operator~(tiny_array_impl<V, OM, N...> const & v)
{
    tiny_array_impl<V, true, N...> res(v.size(), dont_init);
    for(int k=0; k < v.size(); ++k)
        res[k] = ~v[k];
    return res;
}

#define XTENSOR_TINYARRAY_BINARY_FUNCTION(FCT) \
template <class V1, bool OM1, class V2, bool OM2, int ... N> \
inline auto \
FCT(tiny_array_impl<V1, OM1, N...> const & l, \
    tiny_array_impl<V2, OM2, N...> const & r) \
{ \
    XTENSOR_ASSERT_MSG(l.size() == r.size(), #FCT "(tiny_array, tiny_array): size mismatch."); \
    using namespace cmath; \
    tiny_array<decltype(FCT(*(V1*)0, *(V2*)0)), N...> res(l.size(), dont_init); \
    for(int k=0; k < l.size(); ++k) \
        res[k] = FCT(l[k], r[k]); \
    return res; \
}

XTENSOR_TINYARRAY_BINARY_FUNCTION(atan2)
XTENSOR_TINYARRAY_BINARY_FUNCTION(copysign)
XTENSOR_TINYARRAY_BINARY_FUNCTION(fdim)
XTENSOR_TINYARRAY_BINARY_FUNCTION(fmax)
XTENSOR_TINYARRAY_BINARY_FUNCTION(fmin)
XTENSOR_TINYARRAY_BINARY_FUNCTION(fmod)
XTENSOR_TINYARRAY_BINARY_FUNCTION(hypot)

#undef XTENSOR_TINYARRAY_BINARY_FUNCTION

    /** Apply pow() function to each vector component.
    */
template <class V, bool OM, class E, int ... N>
inline auto
pow(tiny_array_impl<V, OM, N...> const & v, E exponent)
{
    using namespace cmath;
    tiny_array<decltype(pow(v[0], exponent)), N...> res(v.size(), dont_init);
    for(int k=0; k < v.size(); ++k)
        res[k] = pow(v[k], exponent);
    return res;
}

    /// cross product
template <class V1, bool OM1, class V2, bool OM2, int N,
          XTENSOR_REQUIRE<N == 3 || N == runtime_size> >
inline
tiny_array<promote_t<V1, V2>, N>
cross(tiny_array_impl<V1, OM1, N> const & r1,
      tiny_array_impl<V2, OM2, N> const & r2)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N, r1.size() == 3 && r2.size() == 3,
        "cross(): cross product requires size() == 3.");
    typedef tiny_array<promote_t<V1, V2>, N> Res;
    return  Res{r1[1]*r2[2] - r1[2]*r2[1],
                r1[2]*r2[0] - r1[0]*r2[2],
                r1[0]*r2[1] - r1[1]*r2[0]};
}

    /// dot product of two vectors
template <class V1, bool OM1, class V2, bool OM2, int N, int M>
inline
promote_t<V1, V2>
dot(tiny_array_impl<V1, OM1, N> const & l,
    tiny_array_impl<V2, OM2, M> const & r)
{
    XTENSOR_ASSERT_MSG(l.size() == r.size(), "dot(): size mismatch.");
    promote_t<V1, V2> res = promote_t<V1, V2>();
    for(int k=0; k < l.size(); ++k)
        res += l[k] * r[k];
    return res;
}

    /// sum of the vector's elements
template <class V, bool OM, int ... N>
inline
promote_t<V>
sum(tiny_array_impl<V, OM, N...> const & l)
{
    promote_t<V> res = promote_t<V>();
    for(int k=0; k < l.size(); ++k)
        res += l[k];
    return res;
}

    /// mean of the vector's elements
template <class V, bool OM, int ... N>
inline real_promote_t<V>
mean(tiny_array_impl<V, OM, N...> const & t)
{
    using Promote = real_promote_t<V>;
    const Promote sumVal = static_cast<Promote>(sum(t));
    if(t.size() > 0)
        return sumVal / static_cast<Promote>(t.size());
    else
        return sumVal;
}

    /// cumulative sum of the vector's elements
template <class V, bool OM, int ... N>
inline
tiny_array_impl<promote_t<V>, true, N...>
cumsum(tiny_array_impl<V, OM, N...> const & l)
{
    tiny_array_impl<promote_t<V>, true, N...> res(l);
    for(int k=1; k < l.size(); ++k)
        res[k] += res[k-1];
    return res;
}

    /// product of the vector's elements
template <class V, bool OM, int ... N>
inline
promote_t<V>
prod(tiny_array_impl<V, OM, N...> const & l)
{
    using Promote = promote_t<V>;
    if(l.size() == 0)
        return Promote();
    Promote res = Promote(1);
    for(int k=0; k < l.size(); ++k)
        res *= l[k];
    return res;
}

    /// cumulative product of the vector's elements
template <class V, bool OM, int ... N>
inline
tiny_array_impl<promote_t<V>, true, N...>
cumprod(tiny_array_impl<V, OM, N...> const & l)
{
    tiny_array_impl<promote_t<V>, true, N...> res(l);
    for(int k=1; k < l.size(); ++k)
        res[k] *= res[k-1];
    return res;
}

    /// element-wise minimum
template <class V1, bool OM1, class V2, bool OM2, int ... N>
inline
tiny_array_impl<promote_t<V1, V2>, true, N...>
min(tiny_array_impl<V1, OM1, N...> const & l,
    tiny_array_impl<V2, OM2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "min(): size mismatch.");
    tiny_array_impl<promote_t<V1, V2>, true, N...> res(l.size(), dont_init);
    for(int k=0; k < l.size(); ++k)
        res[k] =  min(l[k], r[k]);
    return res;
}

    /// element-wise minimum with a constant
template <class V1, bool OM1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value>>
inline
tiny_array_impl<std::common_type_t<V1, V2>, true, N...>
min(tiny_array_impl<V1, OM1, N...> const & l,
    V2 const & r)
{
    tiny_array_impl<std::common_type_t<V1, V2>, true, N...> res(l.size(), dont_init);
    for(int k=0; k < l.size(); ++k)
        res[k] =  min(l[k], r);
    return res;
}

    /// element-wise minimum with a constant
template <class V1, class V2, bool OM2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value>>
inline
tiny_array_impl<std::common_type_t<V1, V2>, true, N...>
min(V1 const & l,
    tiny_array_impl<V2, OM2, N...> const & r)
{
    tiny_array_impl<std::common_type_t<V1, V2>, true, N...> res(r.size(), dont_init);
    for(int k=0; k < r.size(); ++k)
        res[k] =  min(l, r[k]);
    return res;
}

    /** Index of minimal element.

        Returns -1 for an empty array.
    */
template <class V, bool OM, int ... N>
inline int
min_element(tiny_array_impl<V, OM, N...> const & l)
{
    if(l.size() == 0)
        return -1;
    int m = 0;
    for(int i=1; i<l.size(); ++i)
        if(l[i] < l[m])
            m = i;
    return m;
}

    /// minimal element
template <class V, bool OM, int ... N>
inline
V const &
min(tiny_array_impl<V, OM, N...> const & l)
{
    int m = min_element(l);
    xtensor_precondition(m >= 0, "min() on empty tiny_array.");
    return l[m];
}

    /// element-wise maximum
template <class V1, bool OM1, class V2, bool OM2, int ... N>
inline
tiny_array_impl<std::common_type_t<V1, V2>, true, N...>
max(tiny_array_impl<V1, OM1, N...> const & l,
    tiny_array_impl<V2, OM2, N...> const & r)
{
    xtensor_precondition(l.size() == r.size(),
        "max(): size mismatch.");
    tiny_array_impl<std::common_type_t<V1, V2>, true, N...> res(l.size(), dont_init);
    for(int k=0; k < l.size(); ++k)
        res[k] =  max(l[k], r[k]);
    return res;
}

    /// element-wise maximum with a constant
template <class V1, bool OM1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value>>
inline
tiny_array_impl<std::common_type_t<V1, V2>, true, N...>
max(tiny_array_impl<V1, OM1, N...> const & l,
    V2 const & r)
{
    tiny_array_impl<std::common_type_t<V1, V2>, true, N...> res(l.size(), dont_init);
    for(int k=0; k < l.size(); ++k)
        res[k] =  max(l[k], r);
    return res;
}

    /// element-wise maximum with a constant
template <class V1, class V2, bool OM2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value>>
inline
tiny_array_impl<std::common_type_t<V1, V2>, true, N...>
max(V1 const & l,
    tiny_array_impl<V2, OM2, N...> const & r)
{
    tiny_array_impl<std::common_type_t<V1, V2>, true, N...> res(r.size(), dont_init);
    for(int k=0; k < r.size(); ++k)
        res[k] =  max(l, r[k]);
    return res;
}

    /** Index of maximal element.

        Returns -1 for an empty array.
    */
template <class V, bool OM, int ... N>
inline int
max_element(tiny_array_impl<V, OM, N...> const & l)
{
    if(l.size() == 0)
        return -1;
    int m = 0;
    for(int i=1; i<l.size(); ++i)
        if(l[m] < l[i])
            m = i;
    return m;
}

    /// maximal element
template <class V, bool OM, int ... N>
inline V const &
max(tiny_array_impl<V, OM, N...> const & l)
{
    int m = max_element(l);
    xtensor_precondition(m >= 0, "max() on empty tiny_array.");
    return l[m];
}

/// squared norm
template <class V, bool OM, int ... N>
inline squared_norm_t<tiny_array_impl<V, OM, N...> >
squared_norm(tiny_array_impl<V, OM, N...> const & t)
{
    using Type = squared_norm_t<tiny_array_impl<V, OM, N...> >;
    Type result = Type();
    for(int i=0; i<t.size(); ++i)
        result += squared_norm(t[i]);
    return result;
}

template <class V, bool OM, int ... N>
inline
norm_t<V>
mean_square(tiny_array_impl<V, OM, N...> const & t)
{
    return norm_t<V>(squared_norm(t)) / t.size();
}

    /// reversed copy
template <class V, bool OM, int ... N>
inline
tiny_array_impl<V, true, N...>
reversed(tiny_array_impl<V, OM, N...> const & t)
{
    return tiny_array_impl<V, true, N...>(t.begin(), t.end(), copy_reversed);
}

    /** \brief transposed copy

        Elements are arranged such that <tt>res[k] = t[permutation[k]]</tt>.
    */
template <class V1, bool OM1, class V2, bool OM2, int N, int M>
inline
tiny_array<V1, N>
transpose(tiny_array_impl<V1, OM1, N> const & v,
          tiny_array_impl<V2, OM2, M> const & permutation)
{
    return v.transpose(permutation);
}

template <class V1, bool OM1, int N>
inline
tiny_array<V1, N>
transpose(tiny_array_impl<V1, OM1, N> const & v)
{
    return reversed(v);
}

template <class V1, bool OM1, int N1, int N2>
inline
tiny_array<V1, N2, N1>
transpose(tiny_array_impl<V1, OM1, N1, N2> const & v)
{
    tiny_array<V1, N2, N1> res(dont_init);
    for(int i=0; i < N1; ++i)
    {
        for(int j=0; j < N2; ++j)
        {
            res(j,i) = v(i,j);
        }
    }
    return res;
}

    /** \brief Clip negative values.

        All elements smaller than 0 are set to zero.
    */
template <class V, bool OM, int ... N>
inline
tiny_array_impl<V, true, N...>
clip_lower(tiny_array_impl<V, OM, N...> const & t)
{
    return clip_lower(t, V());
}

    /** \brief Clip values below a threshold.

        All elements smaller than \a val are set to \a val.
    */
template <class V, bool OM, int ... N>
inline
tiny_array_impl<V, true, N...>
clip_lower(tiny_array_impl<V, OM, N...> const & t, const V val)
{
    tiny_array_impl<V, true, N...> res(t.size(), dont_init);
    for(int k=0; k < t.size(); ++k)
    {
        res[k] = t[k] < val ? val :  t[k];
    }
    return res;
}

    /** \brief Clip values above a threshold.

        All elements bigger than \a val are set to \a val.
    */
template <class V, bool OM, int ... N>
inline
tiny_array_impl<V, true, N...>
clip_upper(tiny_array_impl<V, OM, N...> const & t, const V val)
{
    tiny_array_impl<V, true, N...> res(t.size(), dont_init);
    for(int k=0; k < t.size(); ++k)
    {
        res[k] = t[k] > val ? val :  t[k];
    }
    return res;
}

    /** \brief Clip values to an interval.

        All elements less than \a valLower are set to \a valLower, all elements
        bigger than \a valUpper are set to \a valUpper.
    */
template <class V, bool OM, int ... N>
inline
tiny_array_impl<V, true, N...>
clip(tiny_array_impl<V, OM, N...> const & t,
     const V valLower, const V valUpper)
{
    tiny_array_impl<V, true, N...> res(t.size(), dont_init);
    for(int k=0; k < t.size(); ++k)
    {
        res[k] =  (t[k] < valLower)
                       ? valLower
                       : (t[k] > valUpper)
                             ? valUpper
                             : t[k];
    }
    return res;
}

    /** \brief Clip values to a vector of intervals.

        All elements less than \a valLower are set to \a valLower, all elements
        bigger than \a valUpper are set to \a valUpper.
    */
template <class V, bool OM1, bool OM2, bool OM3, int ... N>
inline
tiny_array_impl<V, true, N...>
clip(tiny_array_impl<V, OM1, N...> const & t,
     tiny_array_impl<V, OM2, N...> const & valLower,
     tiny_array_impl<V, OM3, N...> const & valUpper)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., t.size() == valLower.size() && t.size() == valUpper.size(),
        "clip(): size mismatch.");
    tiny_array_impl<V, true, N...> res(t.size(), dont_init);
    for(int k=0; k < t.size(); ++k)
    {
        res[k] =  (t[k] < valLower[k])
                       ? valLower[k]
                       : (t[k] > valUpper[k])
                             ? valUpper[k]
                             : t[k];
    }
    return res;
}

template <class T1, bool OM1, class T2, bool OM2, int ... N>
inline void
swap(tiny_array_impl<T1, OM1, N...> & l,
     tiny_array_impl<T2, OM2, N...> & r)
{
    l.swap(r);
}

} // namespace xt

#undef XTENSOR_ASSERT_INSIDE

#endif // XTENSOR_XTINY_HPP
