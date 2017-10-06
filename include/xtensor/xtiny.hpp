/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_XTINY_HPP
#define XTENSOR_XTINY_HPP

#include <array>
#include <type_traits>
#include <iosfwd>
#include <algorithm>
#include <memory>
#include <iterator>
#include <utility>
#include <tuple>  // std::ignore

// #include "xtags.hpp"
#include "xconcepts.hpp"
#include "xexception.hpp"
// #include "xmathutil.hpp"

#ifdef XTENSOR_CHECK_BOUNDS
    #define XTENSOR_ASSERT_INSIDE(array, diff) \
      XTENSOR_PRECONDITION(diff >= 0 && diff < array.size(), "Index out of bounds")
#else
    #define XTENSOR_ASSERT_INSIDE(array, diff)
#endif

namespace xt {

    /// Determine size of an array type at runtime.
static const int runtime_size  = -1;

    /// Don't initialize memory that gets overwritten anyway.
enum skip_initialization_tag { dont_init };

    /// Copy-construct array in reversed order.
enum reverse_copy_tag { copy_reversed };

struct tiny_array_tag {};

template <class T>
struct tiny_array_concept
{
    static const bool value = std::is_base_of<tiny_array_tag, std::decay_t<T>>::value;
};

namespace tags {

enum memory_policy { owns_memory, borrowed_memory };

    // Support for tags::size keyword argument
    // to disambiguate array sizes from initial values.
struct size_proxy
{
    size_t value;
};

struct size_tag
{
    size_proxy operator=(size_t s) const
    {
        return {s};
    }

    size_proxy operator()(size_t s) const
    {
        return {s};
    }
};

namespace {

size_tag size;

}

} // namespace tags

    /** \brief The general type of array indices.

        Note that this is a signed type, so that negative indices
        and index differences work as intuitively expected.
    */
using index_t = std::ptrdiff_t;

template <class VALUETYPE, tags::memory_policy MP, int ... N>
class tiny_array_impl;

template <class VALUETYPE, int M=runtime_size, int ... N>
using tiny_array = tiny_array_impl<VALUETYPE, tags::owns_memory, M, N...>;

template <class VALUETYPE, int M, int N>
using tiny_matrix = tiny_array_impl<VALUETYPE, tags::owns_memory, M, N>;

template <class VALUETYPE, int M=runtime_size, int ... N>
using tiny_array_adaptor = tiny_array_impl<VALUETYPE, tags::borrowed_memory, M, N...>;

namespace detail  {

template<class T>
struct may_use_uninitialized_memory
{
    static const bool value = std::is_scalar<T>::value || std::is_pod<T>::value;
};

template<class T, tags::memory_policy MP, int ... N>
struct may_use_uninitialized_memory<tiny_array_impl<T, MP, N...>>
{
    static const bool value = may_use_uninitialized_memory<T>::value;
};

template<class T, tags::memory_policy MP>
struct may_use_uninitialized_memory<tiny_array_impl<T, MP, runtime_size>>
{
    static const bool value = false;
};

template <index_t LEVEL, int ... N>
struct tiny_storage_helper;

template <index_t LEVEL>
struct tiny_storage_helper<LEVEL, runtime_size>
{};

template <index_t LEVEL, int N, int ... REST>
struct tiny_storage_helper<LEVEL, N, REST...>
{
    static_assert(N >= 0, "tiny_array_impl(): array must have non-negative shape.");
    using next_type = tiny_storage_helper<LEVEL+1, REST...>;

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
struct tiny_storage_helper<LEVEL, N>
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
struct tiny_storage_helper<LEVEL, 0>
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

template <int N0, int ... N>
struct tiny_array_is_static
{
    static const int ndim = sizeof...(N)+1;
    static const bool value = ndim > 1 || N0 != runtime_size;
};

} // namespace detail

#define XTENSOR_ASSERT_RUNTIME_SIZE(SHAPE, PREDICATE, MESSAGE) \
    if(detail::tiny_array_is_static<SHAPE>::value) {} else \
        XTENSOR_PRECONDITION(PREDICATE, MESSAGE)

/********************************************************/
/*                                                      */
/*                    tiny_array_base                   */
/*                                                      */
/********************************************************/

/* \brief Memory management for small arrays.

    This class is normally not used directly. It only provides services related to memory management (constructors, destructors, initialzation, swap), the remaining functionality is implemented in \ref tiny_array_impl and free functions.

    tiny_array_base supports four variants:
    <UL>
        <LI> arrays who own their memory (<tt>MP = tags::owns_memory</tt>) and whose shape is fixed at compile time (<tt>N...</tt> is a number or sequence of numbers)
        <LI> arrays who don't own their memory (<tt>MP = tags::borrowed_memory</tt>) and whose shape is fixed at compile time (<tt>N...</tt> is a number or sequence of numbers)
        <LI> arrays who own their memory (<tt>MP = tags::owns_memory</tt>) and whose shape is determined at runtime (<tt>N == runtime_size</tt>)
        <LI> arrays who don't own their memory (<tt>MP = tags::borrowed_memory</tt>) and whose shape is determined at runtime (<tt>N == runtime_size</tt>)
    </UL>
**/
template <class VALUETYPE, tags::memory_policy MP, int ... N>
class tiny_array_base;

/********************************************************/
/*                                                      */
/*      tiny_array_base: static shape, owns memory      */
/*                                                      */
/********************************************************/

template <class VALUETYPE, int ... N>
class tiny_array_base<VALUETYPE, tags::owns_memory, N...>
: public tiny_array_tag
{
  protected:
    using shape_helper = detail::tiny_storage_helper<0, N...>;
    using data_array_type = VALUETYPE[shape_helper::alloc_size];

    data_array_type data_;

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

    using index_type                  = std::array<index_t, sizeof...(N)>;
    static const bool    is_static    = true;
    static const index_t static_ndim  = sizeof...(N);
    static const index_t static_size  = shape_helper::total_size;
    static const bool may_use_uninitialized_memory =
                                   detail::may_use_uninitialized_memory<VALUETYPE>::value;


    constexpr tiny_array_base(tiny_array_base const &) = default;

    explicit tiny_array_base(skip_initialization_tag)
    {}

    explicit tiny_array_base(VALUETYPE v = VALUETYPE())
    {
        init(v);
    }

    explicit
    tiny_array_base(tags::size_proxy const & size,
                    VALUETYPE const & v = VALUETYPE())
    : tiny_array_base(v)
    {
        std::ignore = size;
        XTENSOR_ASSERT_MSG(size.value == static_size,
            "tiny_array_base(size): size argument conflicts with array length.");
    }

        // for compatibility with tiny_array_impl<..., runtime_size>
    tiny_array_base(tags::size_proxy const & size, skip_initialization_tag)
    {
        std::ignore = size;
        XTENSOR_ASSERT_MSG(size.value == static_size,
            "tiny_array_base(size): size argument conflicts with array length.");
    }

        // for compatibility with tiny_array_impl<..., runtime_size>
    tiny_array_base(index_t size, skip_initialization_tag)
    {
        std::ignore = size;
        XTENSOR_ASSERT_MSG(size == static_size,
            "tiny_array_base(size): size argument conflicts with array length.");
    }

    template <class V, tags::memory_policy OTHER_MP, int ... M>
    tiny_array_base(tiny_array_impl<V, OTHER_MP, M...> const & other)
    {
        if(other.size() == 0)
        {
            init();
        }
        else
        {
            XTENSOR_PRECONDITION(other.size() == static_size,
                "tiny_array_base(tiny_array_base): size mismatch.");
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
    // template <class ... V>
    // constexpr
    // tiny_array_base(VALUETYPE v0, VALUETYPE v1, V ... v)
    // : data_{VALUETYPE(v0), VALUETYPE(v1), VALUETYPE(v)...}
    // {
        // static_assert(sizeof...(V)+2 == static_size,
                      // "tiny_array_base(): number of constructor arguments contradicts size().");
    // }

    template <class V>
    tiny_array_base(std::initializer_list<V> v)
    {
        if(v.size() == 1)
            init(static_cast<VALUETYPE>(*v.begin()));
        else if(v.size() == static_size)
            init_impl(v.begin());
        else
            XTENSOR_PRECONDITION(false,
                "tiny_array_base(std::initializer_list<V>): wrong initialization size (expected: "
                + std::to_string(static_size) + ", got: " + std::to_string(v.size()) +")");
    }

    template <class U>
    explicit tiny_array_base(U const * u, U const *  end  = 0)
    {
        std::ignore = end;
        XTENSOR_ASSERT_MSG(end == 0 || end - u == static_size,
            "tiny_array_base(U const * u, U const * end): size mismatch.");
        init_impl(u);
    }

    template <class ITERATOR,
              XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value> >
    tiny_array_base(ITERATOR u, ITERATOR end)
    {
        std::ignore = end;
        XTENSOR_ASSERT_MSG(std::distance(u, end) == static_size,
            "tiny_array_base(ITERATOR u, ITERATOR end): size mismatch.");
        init_impl(u);
    }

    template <class ITERATOR,
              XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value> >
    tiny_array_base(ITERATOR u, ITERATOR end, reverse_copy_tag)
    {
        XTENSOR_ASSERT_MSG(std::distance(u, end) == static_size,
            "tiny_array_base(ITERATOR u, ITERATOR end, reverse_copy_tag): size mismatch.");
        for(int i=static_size-1; i>=0; --i, ++u)
        {
            data_[i] = static_cast<VALUETYPE>(*u);
        }
    }

    explicit
    tiny_array_base(VALUETYPE const (&u)[1])
    {
        init(*u);
    }

    template <class U>
    explicit tiny_array_base(U const (&u)[1])
    {
        init(static_cast<VALUETYPE>(*u));
    }

    template <class U, int S=static_size,
              XTENSOR_REQUIRE<(S > 1)>>
    explicit tiny_array_base(U const (&u)[static_size])
    {
        init_impl(u);
    }

    constexpr index_t size()  const { return static_size; }
    constexpr index_t max_size()  const { return static_size; }
    constexpr index_type shape() const { return index_type{ N... }; }
    constexpr index_t ndim()  const { return static_ndim; }

    template <class OTHER, tags::memory_policy OTHER_MP>
    constexpr bool
    is_same_shape(tiny_array_impl<OTHER, OTHER_MP, N...> const &) const
    {
        return true;
    }

    template <class OTHER, tags::memory_policy OTHER_MP, int ... M>
    constexpr bool
    is_same_shape(tiny_array_impl<OTHER, OTHER_MP, M...> const &) const
    {
        return false;
    }

    template <class OTHER, tags::memory_policy OTHER_MP>
    bool
    is_same_shape(tiny_array_impl<OTHER, OTHER_MP, runtime_size> const & other) const
    {
        return static_ndim == 1 && this->size() == other.size();
    }

    void init(VALUETYPE const & v = VALUETYPE())
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = v;
    }

    template <class ... V>
    void init(VALUETYPE v0, VALUETYPE v1, V... v)
    {
        static_assert(sizeof...(V)+2 == static_size,
                      "tiny_array_base::init(): wrong number of arguments.");
        init_impl<0>(v0, v1, v...);
    }

    template <class Iterator>
    void init(Iterator first, Iterator end)
    {
        const index_t range = std::distance(first, end);
        if(range == 1)
        {
            init(static_cast<VALUETYPE>(*first));
        }
        else
        {
            XTENSOR_PRECONDITION(range == static_size,
                "tiny_array_base::init(): size mismatch.");
            init_impl(first);
        }
    }

    template <class OTHER, tags::memory_policy OTHER_MP>
    void swap(tiny_array_impl<OTHER, OTHER_MP, N...> & other)
    {
        for(int k=0; k<static_size; ++k)
        {
            VALUETYPE t = data_[k];
            data_[k] = static_cast<VALUETYPE>(other[k]);
            other[k] = static_cast<OTHER>(t);
        }
    }
};

/********************************************************/
/*                                                      */
/*   tiny_array_base: static shape, borrowed memory     */
/*                                                      */
/********************************************************/

template <class VALUETYPE, int ... N>
class tiny_array_base<VALUETYPE, tags::borrowed_memory, N...>
: public tiny_array_tag
{
  protected:
    using shape_helper = detail::tiny_storage_helper<0, N...>;
    using data_array_type = VALUETYPE *;
    data_array_type data_;

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

    using index_type                  = std::array<index_t, sizeof...(N)>;
    static const bool    is_static    = true;
    static const index_t static_ndim  = sizeof...(N);
    static const index_t static_size  = shape_helper::total_size;
    static const bool may_use_uninitialized_memory =
                                   detail::may_use_uninitialized_memory<VALUETYPE>::value;

    constexpr tiny_array_base(tiny_array_base const &) = default;

    explicit tiny_array_base(skip_initialization_tag)
    {}

    explicit tiny_array_base()
    : data_(nullptr)
    {}

        // for compatibility with tiny_array_impl<..., runtime_size>
    tiny_array_base(tags::size_proxy const & size, skip_initialization_tag)
    {
        std::ignore = size;
        XTENSOR_ASSERT_MSG(size.value == static_size,
            "tiny_array_base(size): size argument conflicts with array length.");
    }

        // for compatibility with tiny_array_impl<..., runtime_size>
    tiny_array_base(index_t size, skip_initialization_tag)
    {
        std::ignore = size;
        XTENSOR_ASSERT_MSG(size == static_size,
            "tiny_array_base(size): size argument conflicts with array length.");
    }

    template <tags::memory_policy OTHER_MP>
    explicit
    tiny_array_base(tiny_array_impl<VALUETYPE, OTHER_MP, N...> const & other)
    : data_(const_cast<VALUETYPE *>(other.data()))
    {
        XTENSOR_PRECONDITION(size() == other.size(),
            "tiny_array_base(tiny_array_base): size mismatch.");
    }

        /** Construct view for given pointer
        */
    explicit tiny_array_base(VALUETYPE const * u, VALUETYPE const * end  = 0)
    : data_(const_cast<VALUETYPE *>(u))
    {
        std::ignore = end;
        XTENSOR_ASSERT_MSG(end == 0 || end - u == static_size,
            "tiny_array_base(VALUETYPE const * u, VALUETYPE const * end): size mismatch.");
    }

    constexpr index_t size()  const { return static_size; }
    constexpr index_t max_size()  const { return static_size; }
    constexpr index_type shape() const { return index_type{ N... }; }
    constexpr index_t ndim()  const { return static_ndim; }

    template <class OTHER, tags::memory_policy OTHER_MP>
    constexpr bool
    is_same_shape(tiny_array_impl<OTHER, OTHER_MP, N...> const &) const
    {
        return true;
    }

    template <class OTHER, tags::memory_policy OTHER_MP, int ... M>
    constexpr bool
    is_same_shape(tiny_array_impl<OTHER, OTHER_MP, M...> const &) const
    {
        return false;
    }

    template <class OTHER, tags::memory_policy OTHER_MP>
    bool
    is_same_shape(tiny_array_impl<OTHER, OTHER_MP, runtime_size> const & other) const
    {
        return static_ndim == 1 && this->size() == other.size();
    }

        /** Reset to the other pointer.
        */
    void reset(VALUETYPE const * other)
    {
        data_ = const_cast<VALUETYPE *>(other);
    }

    void init(VALUETYPE const & v = VALUETYPE())
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = v;
    }

    template <class ... V>
    void init(VALUETYPE v0, VALUETYPE v1, V... v)
    {
        static_assert(sizeof...(V)+2 == static_size,
                      "tiny_array_base::init(): wrong number of arguments.");
        init_impl<0>(v0, v1, v...);
    }

    template <class Iterator>
    void init(Iterator first, Iterator end)
    {
        const index_t range = std::distance(first, end);
        if(range == 1)
        {
            init(static_cast<VALUETYPE>(*first));
        }
        else
        {
            XTENSOR_PRECONDITION(range == static_size,
                "tiny_array_base::init(): size mismatch.");
            init_impl(first);
        }
    }

    template <class OTHER, tags::memory_policy OTHER_MP>
    void swap(tiny_array_impl<OTHER, OTHER_MP, N...> & other)
    {
        for(int k=0; k<static_size; ++k)
        {
            VALUETYPE t = data_[k];
            data_[k] = static_cast<VALUETYPE>(other[k]);
            other[k] = static_cast<OTHER>(t);
        }
    }
};

/********************************************************/
/*                                                      */
/*      tiny_array_base: dynamic shape, owns memory     */
/*                                                      */
/********************************************************/

template <class VALUETYPE>
class tiny_array_base<VALUETYPE, tags::owns_memory, runtime_size>
: public tiny_array_tag
{
  protected:
    using shape_helper = detail::tiny_storage_helper<0, runtime_size>;

    // FIXME: implement an optimized allocator
    // FIXME: (look at Alexandrescu's Loki library or Kolmogorov's code)
    std::allocator<VALUETYPE> alloc_;
    index_t size_;
    VALUETYPE * data_;

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
        for(index_t k=0; k < size_; ++k, ++u)
            data_[k] = static_cast<VALUETYPE>(*u);
    }

  public:

    static const bool    is_static    = false;
    static const index_t static_size  = runtime_size;
    static const index_t static_ndim  = 1;
    static const bool may_use_uninitialized_memory =
                                   detail::may_use_uninitialized_memory<VALUETYPE>::value;

    tiny_array_base()
    : size_(0)
    , data_(nullptr)
    {}

    tiny_array_base(tiny_array_base && rhs)
    : tiny_array_base()
    {
        rhs.swap(*this);
    }

    tiny_array_base(tiny_array_base const & rhs )
    : size_(rhs.size())
    , data_(alloc_.allocate(size_))
    {
        std::uninitialized_copy(rhs.data_, rhs.data_+size_, data_);
    }

    explicit
    tiny_array_base(index_t size,
                    VALUETYPE const & initial = VALUETYPE())
    : size_(size)
    , data_(alloc_.allocate(size))
    {
        std::uninitialized_fill(data_, data_+size_, initial);
    }

    explicit
    tiny_array_base(tags::size_proxy const & size,
                    VALUETYPE const & initial = VALUETYPE())
    : tiny_array_base(size.value, initial)
    {}

    tiny_array_base(index_t size, skip_initialization_tag)
    : size_(size)
    , data_(alloc_.allocate(size))
    {
        if(!may_use_uninitialized_memory)
            std::uninitialized_fill(data_, data_+size_, VALUETYPE());
    }

    template <class U, tags::memory_policy OTHER_MP, int ... N>
    tiny_array_base(tiny_array_impl<U, OTHER_MP, N...> const & other)
    : tiny_array_base(other.begin(), other.end())
    {}

    template <class U,
              XTENSOR_REQUIRE<iterator_concept<U>::value> >
    tiny_array_base(U begin, U end)
    : size_(std::distance(begin, end))
    , data_(alloc_.allocate(size_))
    {
        for(int i=0; i<size_; ++i, ++begin)
            new(data_+i) VALUETYPE(static_cast<VALUETYPE>(*begin));
    }

    template <class U,
              XTENSOR_REQUIRE<iterator_concept<U>::value> >
    tiny_array_base(U begin, U end, reverse_copy_tag)
    : size_(std::distance(begin, end))
    , data_(alloc_.allocate(size_))
    {
        for(index_t i=size_-1; i>=0; --i, ++begin)
        {
            new(data_+i) VALUETYPE(static_cast<VALUETYPE>(*begin));
        }
    }

    template <class U, size_t SIZE>
    tiny_array_base(const U (&u)[SIZE])
    : tiny_array_base(u, u+SIZE)
    {}

    template <class U>
    tiny_array_base(std::initializer_list<U> rhs)
    : tiny_array_base(rhs.begin(), rhs.end())
    {}

    ~tiny_array_base()
    {
        if(!may_use_uninitialized_memory)
        {
            for(index_t i=0; i<size_; ++i)
                (data_+i)->~VALUETYPE();
        }
        alloc_.deallocate(data_, size_);
    }

    void resize(size_t new_size)
    {
        if(new_size != size())
        {
            tiny_array_base(new_size).swap(*this);
        }
    }

    index_t size() const { return size_; }
    index_t max_size()  const { return size_; }
    index_t ndim()  const { return static_ndim; }

    template <class OTHER, tags::memory_policy OTHER_MP, int ... M>
    bool
    is_same_shape(tiny_array_impl<OTHER, OTHER_MP, M...> const & other) const
    {
        return sizeof...(M) == 1 && this->size() == other.size();
    }

    void init(VALUETYPE const & v = VALUETYPE())
    {
        for(int i=0; i<size_; ++i)
            data_[i] = v;
    }

    template <class ... V>
    void init(VALUETYPE v0, VALUETYPE v1, V... v)
    {
        XTENSOR_PRECONDITION(sizeof...(V)+2 == size_,
                      "tiny_array_base::init(): wrong number of arguments.");
        init_impl<0>(v0, v1, v...);
    }

    template <class ITERATOR,
              XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value>>
    void init(ITERATOR first, ITERATOR end)
    {
        index_t range = std::distance(first, end);
        if(range == 1)
            init(static_cast<VALUETYPE>(*first));
        else if(range == size_)
            init_impl(first);
        else
            XTENSOR_PRECONDITION(false,
                "tiny_array_base::init(): size mismatch.");
    }

    template <class V>
    void init(std::initializer_list<V> l)
    {
        init(l.begin(), l.end());
    }

    void swap(tiny_array_base & other)
    {
        using std::swap;
        swap(this->size_, other.size_);
        swap(this->data_, other.data_);
    }
};

/********************************************************/
/*                                                      */
/*   tiny_array_base: dynamic shape, borrowed memory    */
/*                                                      */
/********************************************************/

template <class VALUETYPE>
class tiny_array_base<VALUETYPE, tags::borrowed_memory, runtime_size>
: public tiny_array_tag
{
  protected:
    using shape_helper = detail::tiny_storage_helper<0, runtime_size>;

    index_t size_;
    VALUETYPE * data_;

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
        for(index_t k=0; k < size_; ++k, ++u)
            data_[k] = static_cast<VALUETYPE>(*u);
    }

  public:

    static const bool    is_static    = false;
    static const index_t static_size  = runtime_size;
    static const index_t static_ndim  = 1;
    static const bool may_use_uninitialized_memory =
                                   detail::may_use_uninitialized_memory<VALUETYPE>::value;

    tiny_array_base()
    : size_(0)
    , data_(nullptr)
    {}

    tiny_array_base(tiny_array_base const & rhs )
    : size_(rhs.size_)
    , data_(const_cast<VALUETYPE*>(rhs.data_))
    {}

    tiny_array_base(index_t size, VALUETYPE const * data)
    : size_(size)
    , data_(const_cast<VALUETYPE*>(data))
    {}

    index_t size() const { return size_; }
    index_t max_size()  const { return size_; }
    index_t ndim()  const { return static_ndim; }

    template <class OTHER, tags::memory_policy OTHER_MP, int ... M>
    bool
    is_same_shape(tiny_array_impl<OTHER, OTHER_MP, M...> const & other) const
    {
        return sizeof...(M) == 1 && this->size() == other.size();
    }

        /** Reset to the other pointer.
        */
    void reset(index_t size, VALUETYPE const * other)
    {
        size_ = size;
        data_ = const_cast<VALUETYPE *>(other);
    }

    void init(VALUETYPE const & v = VALUETYPE())
    {
        for(int i=0; i<size_; ++i)
            data_[i] = v;
    }

    template <class ... V>
    void init(VALUETYPE v0, VALUETYPE v1, V... v)
    {
        XTENSOR_PRECONDITION(sizeof...(V)+2 == size_,
                      "tiny_array_base::init(): wrong number of arguments.");
        init_impl<0>(v0, v1, v...);
    }

    template <class ITERATOR,
              XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value>>
    void init(ITERATOR first, ITERATOR end)
    {
        index_t range = std::distance(first, end);
        if(range == 1)
            init(static_cast<VALUETYPE>(*first));
        else if(range == size_)
            init_impl(first);
        else
            XTENSOR_PRECONDITION(false,
                "tiny_array_base::init(): size mismatch.");
    }

    template <class V>
    void init(std::initializer_list<V> l)
    {
        init(l.begin(), l.end());
    }

    void swap(tiny_array_base & other)
    {
        using std::swap;
        swap(this->size_, other.size_);
        swap(this->data_, other.data_);
    }
};

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
template <class VALUETYPE, tags::memory_policy MP, int ... N>
class tiny_array_impl
: public tiny_array_base<VALUETYPE, MP, N ...>
{
    using tiny_array_base<VALUETYPE, MP, N ...>::shape_helper;

  public:
    template <class NEW_VALUETYPE>
    using as_type = tiny_array_impl<NEW_VALUETYPE, MP, N...>;

    using base_type              = tiny_array_base<VALUETYPE, MP, N ...>;
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
    using index_type             = std::array<index_t, sizeof...(N)>;

    using base_type::is_static;
    using base_type::static_ndim;
    using base_type::static_size;
    using base_type::may_use_uninitialized_memory;

    using base_type::base_type;

    tiny_array_impl()
    : base_type()
    {}

    tiny_array_impl(tiny_array_impl const & rhs)
    : base_type(rhs)
    {}

    tiny_array_impl(tiny_array_impl && rhs)
    : base_type(std::forward<base_type>(rhs))
    {}

    // assignment

    tiny_array_impl & operator=(value_type const & v)
    {
        this->init(v);
        return *this;
    }

    tiny_array_impl & operator=(tiny_array_impl const & rhs)
    {
        if(this == &rhs)
            return *this;
        if(this->size() != rhs.size())
        {
            // can only happen if static_size == runtime_size
            tiny_array_impl(rhs).swap(*this);
        }
        else
        {
            init(rhs.begin(), rhs.end());
        }
        return *this;
    }

    tiny_array_impl & operator=(tiny_array_impl && rhs)
    {
        if(this->size() != rhs.size())
            rhs.swap(*this);
        else
            this->init_impl(rhs.begin());
        return *this;
    }

    template<int M>
    tiny_array_impl & operator=(value_type const (&v)[M])
    {
        if(this->size() != M)
        {
            XTENSOR_PRECONDITION(!is_static && MP==tags::owns_memory,
                "tiny_array_impl::operator=(): size mismatch.");
            tiny_array_impl(v, v+M).swap(*this);
        }
        else
        {
            this->init_impl(v);
        }
        return *this;
    }

    template <class U, tags::memory_policy OTHER_MP, int ... M>
    tiny_array_impl & operator=(tiny_array_impl<U, OTHER_MP, M...> const & rhs)
    {
        if(this->size() != rhs.size())
        {
            XTENSOR_PRECONDITION(!is_static && MP==tags::owns_memory,
                "tiny_array_impl::operator=(): size mismatch.");
            tiny_array_impl(rhs).swap(*this);
        }
        else
        {
            this->init_impl(rhs.begin());
        }
        return *this;
    }

    // index access

    reference operator[](index_t i)
    {
        return this->data_[i];
    }

    constexpr const_reference operator[](index_t i) const
    {
        return this->data_[i];
    }

    reference at(index_t i)
    {
        if(i < 0 || i >= this->size())
            throw std::out_of_range("tiny_array_impl::at()");
        return this->data_[i];
    }

    const_reference at(index_t i) const
    {
        if(i < 0 || i >= this->size())
            throw std::out_of_range("tiny_array_impl::at()");
        return this->data_[i];
    }

    template <bool IS_STATIC=is_static>
    std::enable_if_t<IS_STATIC, reference>
    operator[](index_t const (&i)[static_ndim])
    {
        return this->data_[shape_helper::offset(i)];
    }

    template <bool IS_STATIC=is_static>
    std::enable_if_t<IS_STATIC, const_reference>
    operator[](index_t const (&i)[static_ndim]) const
    {
        return this->data_[shape_helper::offset(i)];
    }

    template <bool IS_STATIC=is_static>
    std::enable_if_t<IS_STATIC, reference>
    at(index_t const (&i)[static_ndim])
    {
        return at(shape_helper::offset(i));
    }

    template <bool IS_STATIC=is_static>
    std::enable_if_t<IS_STATIC, const_reference>
    at(index_t const (&i)[static_ndim]) const
    {
        return at(shape_helper::offset(i));
    }

    template <bool IS_STATIC=is_static>
    std::enable_if_t<IS_STATIC, reference>
    operator[](index_type const & i)
    {
        return this->data_[shape_helper::offset(i.data())];
    }

    template <bool IS_STATIC=is_static>
    std::enable_if_t<IS_STATIC, reference>
    operator[](index_type const & i) const
    {
        return this->data_[shape_helper::offset(i.data())];
    }

    template <bool IS_STATIC=is_static>
    std::enable_if_t<IS_STATIC, reference>
    at(index_type const & i)
    {
        return at(shape_helper::offset(i.data()));
    }

    template <bool IS_STATIC=is_static>
    std::enable_if_t<IS_STATIC, const_reference>
    at(index_type const & i) const
    {
        return at(shape_helper::offset(i.data()));
    }

    template <class ... V>
    reference operator()(V...v)
    {
        static_assert(sizeof...(V) == static_ndim,
                      "tiny_array_impl::operator(): wrong number of arguments.");
        return this->data_[shape_helper::offset(v...)];
    }

    template <class ... V>
    constexpr const_reference operator()(V...v) const
    {
        static_assert(sizeof...(V) == static_ndim,
                      "tiny_array_impl::operator(): wrong number of arguments.");
        return this->data_[shape_helper::offset(v...)];
    }

        /** Get a view to the subarray with length <tt>(TO-FROM)</tt> starting at <tt>FROM</tt>.
            The bounds must fullfill <tt>0 <= FROM < TO <= size()</tt>.
            Only available if this array is 1-dimensional, i.e. <tt>static_ndim == 1</tt>.
        */
    template <int FROM, int TO>
    tiny_array_adaptor<value_type, TO-FROM>
    subarray() const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array_impl::subarray(): array must be 1-dimensional.");
        static_assert(FROM >= 0 && FROM < TO,
            "tiny_array_impl::subarray(): range out of bounds.");
        XTENSOR_PRECONDITION(TO <= this->size(),
            "tiny_array_impl::subarray(): range out of bounds.");
        return tiny_array_adaptor<value_type, TO-FROM>(this->data_+FROM);
    }

        /** Get a view to the subarray with length <tt>(TO-FROM)</tt> starting at <tt>FROM</tt>.
            The bounds must fullfill <tt>0 <= FROM < TO <= size()</tt>.
            Only available if this array is 1-dimensional, i.e. <tt>static_ndim == 1</tt>.
        */
    tiny_array_adaptor<value_type, runtime_size>
    subarray(index_t FROM, index_t TO) const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array_impl::subarray(): array must be 1-dimensional.");
        XTENSOR_PRECONDITION(FROM >= 0 && FROM < TO && TO <= this->size(),
            "tiny_array_impl::subarray(): range out of bounds.");
        return tiny_array_adaptor<value_type, runtime_size>(TO-FROM, this->data_+FROM);
    }

    tiny_array<value_type, is_static ? static_size-1 : runtime_size>
    erase(index_t m) const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array_impl::erase(): array must be 1-dimensional.");
        XTENSOR_PRECONDITION(m >= 0 && m < this->size(), "tiny_array::erase(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(this->size())+").");
        static const index_t res_size = is_static
                                            ? static_size-1
                                            : runtime_size;
        tiny_array<value_type, res_size> res(this->size()-1, dont_init);
        for(int k=0; k<m; ++k)
            res[k] = this->data_[k];
        for(index_t k=m+1; k<this->size(); ++k)
            res[k-1] = this->data_[k];
        return res;
    }

    tiny_array<value_type, is_static ? static_size-1 : runtime_size>
    pop_front() const
    {
        return erase(0);
    }

    tiny_array<value_type, is_static ? static_size-1 : runtime_size>
    pop_back() const
    {
        return erase(this->size()-1);
    }

    tiny_array<value_type, is_static ? static_size+1 : runtime_size>
    insert(index_t m, value_type v) const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array_impl::insert(): array must be 1-dimensional.");
        XTENSOR_PRECONDITION(m >= 0 && m <= this->size(), "tiny_array::insert(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(this->size())+"].");
        static const index_t res_size = is_static
                                            ? static_size+1
                                            : runtime_size;
        tiny_array<value_type, res_size> res(this->size()+1, dont_init);
        for(int k=0; k<m; ++k)
            res[k] = this->data_[k];
        res[m] = v;
        for(index_t k=m; k<this->size(); ++k)
            res[k+1] = this->data_[k];
        return res;
    }

    template <class V, tags::memory_policy MP, int M>
    inline
    tiny_array<value_type, static_size>
    transpose(tiny_array_impl<V, MP, M> const & permutation) const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array::transpose(): only allowed for 1-dimensional arrays.");
        static_assert(M == static_size || M == runtime_size,
            "tiny_array::transpose(): size mismatch.");
        XTENSOR_PRECONDITION(this->size() == permutation.size(),
            "tiny_array::transpose(): size mismatch.");
        tiny_array<value_type, static_size> res(this->size(), dont_init);
        for(int k=0; k < this->size(); ++k)
        {
            XTENSOR_ASSERT_MSG(permutation[k] >= 0 && permutation[k] < this->size(),
                "tiny_array::transpose():  Permutation index out of bounds");
            res[k] = (*this)[permutation[k]];
        }
        return res;
    }

    // boiler plate

    iterator begin() { return this->data_; }
    iterator end()   { return this->data_ + this->size(); }
    const_iterator begin() const { return this->data_; }
    const_iterator end()   const { return this->data_ + this->size(); }
    const_iterator cbegin() const { return this->data_; }
    const_iterator cend()   const { return this->data_ + this->size(); }

    reverse_iterator rbegin() { return reverse_iterator(this->data_ + this->size()); }
    reverse_iterator rend()   { return reverse_iterator(this->data_); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(this->data_ + this->size()); }
    const_reverse_iterator rend()   const { return const_reverse_iterator(this->data_); }
    const_reverse_iterator crbegin() const { return const_reverse_iterator(this->data_ + this->size()); }
    const_reverse_iterator crend()   const { return const_reverse_iterator(this->data_); }

    pointer data() { return this->data_; }
    const_pointer data() const { return this->data_; }

    reference front() { return this->data_[0]; }
    reference back()  { return this->data_[this->size()-1]; }
    constexpr const_reference front() const { return this->data_[0]; }
    constexpr const_reference back()  const { return this->data_[this->size()-1]; }

    constexpr bool    empty() const { return this->size() == 0; }

    tiny_array_impl & reverse()
    {
        using std::swap;
        index_t i=0, j=this->size()-1;
        while(i < j)
             swap(this->data_[i++], this->data_[j--]);
        return *this;
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
        static_assert(SIZE > 0,
            "tiny_array::unit_vector(): SIZE must be poisitive.");
        tiny_array<value_type, SIZE> res;
        res[k] = 1;
        return res;
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

        /// factory function for fixed-size linear sequence ending at <tt>end-1</tt>
    static inline
    tiny_array_impl<value_type, tags::owns_memory, N...>
    range(value_type end)
    {
        XTENSOR_PRECONDITION(static_size != runtime_size || end >= 0,
            "tiny_array::range(): end must be non-negative.");
        value_type start = (static_size != runtime_size)
                               ? end - static_cast<value_type>(static_size)
                               : value_type();
        tiny_array_impl<value_type, tags::owns_memory, N...> res(tags::size = end-start, dont_init);
        for(index_t k=0; k < res.size(); ++k, ++start)
            res[k] = start;
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
        XTENSOR_PRECONDITION(step != 0,
            "tiny_array::range(): step must be non-zero.");
        XTENSOR_PRECONDITION((step > 0 && begin <= end) || (step < 0 && begin >= end),
            "tiny_array::range(): sign mismatch between step and (end-begin).");
        index_t size = floor((abs(end-begin+step)-1)/abs(step));
        tiny_array<value_type, runtime_size> res(size, dont_init);
        for(index_t k=0; k < size; ++k, begin += step)
            res[k] = begin;
        return res;
    }

        /// factory function for fixed-size linear sequence starting at <tt>start</tt> with stepsize <tt>step</tt>
    template <int SIZE=static_size>
    static inline
    tiny_array<value_type, SIZE>
    linear_sequence(value_type start = value_type(), value_type step = value_type(1))
    {
        static_assert(SIZE > 0,
            "tiny_array::linear_sequence(): SIZE must be poisitive.");
        tiny_array_impl<value_type, tags::owns_memory, SIZE> res(dont_init);
        for(index_t k=0; k < SIZE; ++k, start += step)
            res[k] = start;
        return res;
    }
};

/********************************************************/
/*                                                      */
/*                tiny_array_impl output                */
/*                                                      */
/********************************************************/

template <class T, tags::memory_policy MP, int ... N>
std::ostream & operator<<(std::ostream & o, tiny_array_impl<T, MP, N...> const & v)
{
    o << "{";
    if(v.size() > 0)
        o << promote_type_t<T>(v[0]);
    for(int i=1; i < v.size(); ++i)
        o << ", " << promote_type_t<T>(v[i]);
    o << "}";
    return o;
}

template <class T, tags::memory_policy MP, int N1, int N2>
std::ostream & operator<<(std::ostream & o, tiny_array_impl<T, MP, N1, N2> const & v)
{
    o << "{";
    for(int i=0; N2>0 && i<N1; ++i)
    {
        if(i > 0)
            o << ",\n ";
        o << promote_type_t<T>(v(i,0));
        for(int j=1; j<N2; ++j)
        {
            o << ", " << promote_type_t<T>(v(i, j));
        }
    }
    o << "}";
    return o;
}

/********************************************************/
/*                                                      */
/*                tiny_array Comparison                 */
/*                                                      */
/********************************************************/

/** \addtogroup TinyArrayOperators Functions for tiny_array

    \brief Implement basic arithmetic and equality for tiny_array.

    These functions fulfill the requirements of a Linear Space (vector space).
    Return types are determined according to \ref promote_type_t or \ref real_promote_type_t.
*/
//@{

    /// element-wise equal
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ...M, int ... N>
inline bool
operator==(tiny_array_impl<V1, MP1, M...> const & l,
           tiny_array_impl<V2, MP2, N...> const & r)
{
    if(!l.is_same_shape(r))
        return false;
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r[k])
            return false;
    return true;
}

    /// element-wise equal to a constant
template <class V1, tags::memory_policy MP1, class V2, int ...M,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
operator==(tiny_array_impl<V1, MP1, M...> const & l,
           V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r)
            return false;
    return true;
}

    /// element-wise equal to a constant
template <class V1, class V2, tags::memory_policy MP2, int ...M,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
operator==(V1 const & l,
           tiny_array_impl<V2, MP2, M...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if(l != r[k])
            return false;
    return true;
}

    /// element-wise not equal
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... M, int ... N>
inline bool
operator!=(tiny_array_impl<V1, MP1, M...> const & l,
           tiny_array_impl<V2, MP2, N...> const & r)
{
    if(!l.is_same_shape(r))
        return true;
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r[k])
            return true;
    return false;
}

    /// element-wise not equal to a constant
template <class V1, tags::memory_policy MP1, class V2, int ... M,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
operator!=(tiny_array_impl<V1, MP1, M...> const & l,
           V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r)
            return true;
    return false;
}

    /// element-wise not equal to a constant
template <class V1, class V2, tags::memory_policy MP2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
operator!=(V1 const & l,
           tiny_array_impl<V2, MP2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if(l != r[k])
            return true;
    return false;
}

    /// lexicographical less
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
inline bool
operator<(tiny_array_impl<V1, MP1, N...> const & l,
          tiny_array_impl<V2, MP2, N...> const & r)
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
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
inline bool
operator<=(tiny_array_impl<V1, MP1, N...> const & l,
           tiny_array_impl<V2, MP2, N...> const & r)
{
    return !(r < l);
}

    /// lexicographical greater
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
inline bool
operator>(tiny_array_impl<V1, MP1, N...> const & l,
          tiny_array_impl<V2, MP2, N...> const & r)
{
    return r < l;
}

    /// lexicographical greater-equal
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
inline bool
operator>=(tiny_array_impl<V1, MP1, N...> const & l,
           tiny_array_impl<V2, MP2, N...> const & r)
{
    return !(l < r);
}

    /// check if all elements are non-zero (or 'true' if V is bool)
template <class V, tags::memory_policy MP, int ... N>
inline bool
all(tiny_array_impl<V, MP, N...> const & t)
{
    for(int i=0; i<t.size(); ++i)
        if(t[i] == V())
            return false;
    return true;
}

    /// check if at least one element is non-zero (or 'true' if V is bool)
template <class V, tags::memory_policy MP, int ... N>
inline bool
any(tiny_array_impl<V, MP, N...> const & t)
{
    for(int i=0; i<t.size(); ++i)
        if(t[i] != V())
            return true;
    return false;
}

    /// check if all elements are zero (or 'false' if V is bool)
template <class V, tags::memory_policy MP, int ... N>
inline bool
all_zero(tiny_array_impl<V, MP, N...> const & t)
{
    for(int i=0; i<t.size(); ++i)
        if(t[i] != V())
            return false;
    return true;
}

    /// pointwise less-than
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
inline bool
all_less(tiny_array_impl<V1, MP1, N...> const & l,
         tiny_array_impl<V2, MP2, N...> const & r)
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
template <class V1, tags::memory_policy MP1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_less(tiny_array_impl<V1, MP1, N...> const & l,
         V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] >= r)
            return false;
    return true;
}

    /// constant pointwise less than the vector
    /// (typically used to check positivity with `l = 0`)
template <class V1, class V2, tags::memory_policy MP2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_less(V1 const & l,
         tiny_array_impl<V2, MP2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l >= r[k])
            return false;
    return true;
}

    /// pointwise less-equal
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
inline bool
all_less_equal(tiny_array_impl<V1, MP1, N...> const & l,
               tiny_array_impl<V2, MP2, N...> const & r)
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
template <class V1, tags::memory_policy MP1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_less_equal(tiny_array_impl<V1, MP1, N...> const & l,
               V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] > r)
            return false;
    return true;
}

    /// pointwise less-equal with a constant
    /// (typically used to check non-negativity with `l = 0`)
template <class V1, class V2, tags::memory_policy MP2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_less_equal(V1 const & l,
               tiny_array_impl<V2, MP2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l > r[k])
            return false;
    return true;
}

    /// pointwise greater-than
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
inline bool
all_greater(tiny_array_impl<V1, MP1, N...> const & l,
            tiny_array_impl<V2, MP2, N...> const & r)
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
template <class V1, tags::memory_policy MP1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_greater(tiny_array_impl<V1, MP1, N...> const & l,
            V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] <= r)
            return false;
    return true;
}

    /// constant pointwise greater-than a vector
    /// (typically used to check negativity with `l = 0`)
template <class V1, class V2, tags::memory_policy MP2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_greater(V1 const & l,
            tiny_array_impl<V2, MP2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l <= r[k])
            return false;
    return true;
}

    /// pointwise greater-equal
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
inline bool
all_greater_equal(tiny_array_impl<V1, MP1, N...> const & l,
                  tiny_array_impl<V2, MP2, N...> const & r)
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
template <class V1, tags::memory_policy MP1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_greater_equal(tiny_array_impl<V1, MP1, N...> const & l,
                  V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] < r)
            return false;
    return true;
}

    /// pointwise greater-equal with a constant
    /// (typically used to check non-positivity with `l = 0`)
template <class V1, class V2, tags::memory_policy MP2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_greater_equal(V1 const & l,
                  tiny_array_impl<V2, MP2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l < r[k])
            return false;
    return true;
}

template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
inline bool
isclose(tiny_array_impl<V1, MP1, N...> const & l,
        tiny_array_impl<V2, MP2, N...> const & r,
        promote_type_t<V1, V2> epsilon = 2.0*std::numeric_limits<promote_type_t<V1, V2> >::epsilon())
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
template <class V1, tags::memory_policy MP, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
tiny_array_impl<V1, MP, N...> &
operator+=(tiny_array_impl<V1, MP, N...> & l,
           V2 r);

    /// element-wise add-assignment
template <class V1, tags::memory_policy MP, class V2, tags::memory_policy OTHER_MP, int ... N>
tiny_array_impl<V1, MP, N...> &
operator+=(tiny_array_impl<V1, MP, N...> & l,
           tiny_array_impl<V2, OTHER_MP, N...> const & r);

    /// element-wise addition
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
tiny_array<promote_type_t<V1, V2>, N...>
operator+(tiny_array_impl<V1, MP1, N...> const & l,
          tiny_array_impl<V2, MP2, N...> const & r);

    /// element-wise scalar addition
template <class V1, tags::memory_policy MP1, class V2, int ... N>
tiny_array<promote_type_t<V1, V2>, N...>
operator+(tiny_array_impl<V1, MP1, N...> const & l,
          V2 r);

    /// element-wise left scalar addition
template <class V1, class V2, tags::memory_policy MP2, int ... N>
tiny_array<promote_type_t<V1, V2>, N...>
operator+(V1 l,
          tiny_array_impl<V2, MP2, N...> const & r);

    /// scalar subtract-assignment
template <class V1, tags::memory_policy MP, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
tiny_array_impl<V1, MP, N...> &
operator-=(tiny_array_impl<V1, MP, N...> & l,
           V2 r);

    /// element-wise subtract-assignment
template <class V1, tags::memory_policy MP, class V2, tags::memory_policy OTHER_MP, int ... N>
tiny_array_impl<V1, MP, N...> &
operator-=(tiny_array_impl<V1, MP, N...> & l,
           tiny_array_impl<V2, OTHER_MP, N...> const & r);

    /// element-wise subtraction
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
tiny_array<promote_type_t<V1, V2>, N...>
operator-(tiny_array_impl<V1, MP1, N...> const & l,
          tiny_array_impl<V2, MP2, N...> const & r);

    /// element-wise scalar subtraction
template <class V1, tags::memory_policy MP1, class V2, int ... N>
tiny_array<promote_type_t<V1, V2>, N...>
operator-(tiny_array_impl<V1, MP1, N...> const & l,
          V2 r);

    /// element-wise left scalar subtraction
template <class V1, class V2, tags::memory_policy MP2, int ... N>
tiny_array<promote_type_t<V1, V2>, N...>
operator-(V1 l,
          tiny_array_impl<V2, MP2, N...> const & r);

    /// scalar multiply-assignment
template <class V1, tags::memory_policy MP, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
tiny_array_impl<V1, MP, N...> &
operator*=(tiny_array_impl<V1, MP, N...> & l,
           V2 r);

    /// element-wise multiply-assignment
template <class V1, tags::memory_policy MP, class V2, tags::memory_policy OTHER_MP, int ... N>
tiny_array_impl<V1, MP, N...> &
operator*=(tiny_array_impl<V1, MP, N...> & l,
           tiny_array_impl<V2, OTHER_MP, N...> const & r);

    /// element-wise multiplication
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
tiny_array<promote_type_t<V1, V2>, N...>
operator*(tiny_array_impl<V1, MP1, N...> const & l,
          tiny_array_impl<V2, MP2, N...> const & r);

    /// element-wise scalar multiplication
template <class V1, tags::memory_policy MP1, class V2, int ... N>
tiny_array<promote_type_t<V1, V2>, N...>
operator*(tiny_array_impl<V1, MP1, N...> const & l,
          V2 r);

    /// element-wise left scalar multiplication
template <class V1, class V2, tags::memory_policy MP2, int ... N>
tiny_array<promote_type_t<V1, V2>, N...>
operator*(V1 l,
          tiny_array_impl<V2, MP2, N...> const & r);

    /// scalar divide-assignment
template <class V1, tags::memory_policy MP, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
tiny_array_impl<V1, MP, N...> &
operator/=(tiny_array_impl<V1, MP, N...> & l,
           V2 r);

    /// element-wise divide-assignment
template <class V1, tags::memory_policy MP, class V2, tags::memory_policy OTHER_MP, int ... N>
tiny_array_impl<V1, MP, N...> &
operator/=(tiny_array_impl<V1, MP, N...> & l,
           tiny_array_impl<V2, OTHER_MP, N...> const & r);

    /// element-wise division
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
tiny_array<promote_type_t<V1, V2>, N...>
operator/(tiny_array_impl<V1, MP1, N...> const & l,
          tiny_array_impl<V2, MP2, N...> const & r);

    /// element-wise scalar division
template <class V1, tags::memory_policy MP1, class V2, int ... N>
tiny_array<promote_type_t<V1, V2>, N...>
operator/(tiny_array_impl<V1, MP1, N...> const & l,
          V2 r);

    /// element-wise left scalar division
template <class V1, class V2, tags::memory_policy MP2, int ... N>
tiny_array<promote_type_t<V1, V2>, N...>
operator/(V1 l,
          tiny_array_impl<V2, MP2, N...> const & r);

    /// scalar modulo-assignment
template <class V1, tags::memory_policy MP, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
tiny_array_impl<V1, MP, N...> &
operator%=(tiny_array_impl<V1, MP, N...> & l,
           V2 r);

    /// element-wise modulo-assignment
template <class V1, tags::memory_policy MP, class V2, tags::memory_policy OTHER_MP, int ... N>
tiny_array_impl<V1, MP, N...> &
operator%=(tiny_array_impl<V1, MP, N...> & l,
           tiny_array_impl<V2, OTHER_MP, N...> const & r);

    /// element-wise modulo
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
tiny_array<promote_type_t<V1, V2>, N...>
operator%(tiny_array_impl<V1, MP1, N...> const & l,
          tiny_array_impl<V2, MP2, N...> const & r);

    /// element-wise scalar modulo
template <class V1, tags::memory_policy MP1, class V2, int ... N>
tiny_array<promote_type_t<V1, V2>, N...>
operator%(tiny_array_impl<V1, MP1, N...> const & l,
          V2 r);

    /// element-wise left scalar modulo
template <class V1, class V2, tags::memory_policy MP2, int ... N>
tiny_array<promote_type_t<V1, V2>, N...>
operator%(V1 l,
          tiny_array_impl<V2, MP2, N...> const & r);

#else

#define XTENSOR_TINYARRAY_OPERATORS(OP) \
template <class V1, tags::memory_policy MP, int ... N, class V2, \
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value && \
                          std::is_convertible<V2, V1>::value> > \
inline tiny_array_impl<V1, MP, N...> & \
operator OP##=(tiny_array_impl<V1, MP, N...> & l, \
               V2 r) \
{ \
    for(int i=0; i<l.size(); ++i) \
        l[i] OP##= r; \
    return l; \
} \
 \
template <class V1, tags::memory_policy MP, class V2, tags::memory_policy OTHER_MP, int ... N> \
inline tiny_array_impl<V1, MP, N...> &  \
operator OP##=(tiny_array_impl<V1, MP, N...> & l, \
               tiny_array_impl<V2, OTHER_MP, N...> const & r) \
{ \
    XTENSOR_ASSERT_MSG(l.size() == r.size(), \
        "tiny_array_impl::operator" #OP "=(): size mismatch."); \
    for(int i=0; i<l.size(); ++i) \
        l[i] OP##= r[i]; \
    return l; \
} \
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N> \
inline \
tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), tags::owns_memory, N...> \
operator OP(tiny_array_impl<V1, MP1, N...> const & l, \
            tiny_array_impl<V2, MP2, N...> const & r) \
{ \
    tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), tags::owns_memory, N...> res(l); \
    return res OP##= r; \
} \
 \
template <class V1, tags::memory_policy MP1, class V2, int ... N, \
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value> >\
inline \
tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), tags::owns_memory, N...> \
operator OP(tiny_array_impl<V1, MP1, N...> const & l, \
            V2 r) \
{ \
    tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), tags::owns_memory, N...> res(l); \
    return res OP##= r; \
} \
 \
template <class V1, class V2, tags::memory_policy MP2, int ... N, \
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value && \
                          !std::is_base_of<std::ios_base, V1>::value> >\
inline \
tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), tags::owns_memory, N...> \
operator OP(V1 l, \
             tiny_array_impl<V2, MP2, N...> const & r) \
{ \
    tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), tags::owns_memory, N...> res{l}; \
    return res OP##= r; \
} \
 \
template <class V1, class V2, tags::memory_policy MP2, \
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value && \
                          !std::is_base_of<std::ios_base, V1>::value> >\
inline \
tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), tags::owns_memory, runtime_size> \
operator OP(V1 l, \
             tiny_array_impl<V2, MP2, runtime_size> const & r) \
{ \
    tiny_array_impl<decltype((*(V1*)0) OP (*(V2*)0)), tags::owns_memory, runtime_size> res(tags::size=r.size(), l); \
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
template <class V, tags::memory_policy MP, int ... N>
inline tiny_array_impl<real_promote_type_t<V>, tags::owns_memory, N...>
sqrt(tiny_array_impl<V, MP, N...> const & v)
{
    using namespace cmath;
    tiny_array_impl<real_promote_type_t<V>, tags::owns_memory, N...> res(v.size(), dont_init);
    for(int k=0; k < v.size(); ++k)
        res[k] = sqrt(v[k]);
    return res;
}

#define XTENSOR_TINYARRAY_UNARY_FUNCTION(FCT) \
template <class V, tags::memory_policy MP, int ... N> \
inline auto \
FCT(tiny_array_impl<V, MP, N...> const & v) \
{ \
    using namespace cmath; \
    tiny_array<bool_promote_type_t<decltype(FCT(*(V*)0))>, N...> res(v.size(), dont_init); \
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
template <class V, tags::memory_policy MP, int ... N>
inline
tiny_array_impl<V, tags::owns_memory, N...>
operator-(tiny_array_impl<V, MP, N...> const & v)
{
    tiny_array_impl<V, tags::owns_memory, N...> res(v.size(), dont_init);
    for(int k=0; k < v.size(); ++k)
        res[k] = -v[k];
    return res;
}

    /// Boolean negation
template <class V, tags::memory_policy MP, int ... N>
inline
tiny_array_impl<V, tags::owns_memory, N...>
operator!(tiny_array_impl<V, MP, N...> const & v)
{
    tiny_array_impl<V, tags::owns_memory, N...> res(v.size(), dont_init);
    for(int k=0; k < v.size(); ++k)
        res[k] = !v[k];
    return res;
}

    /// Bitwise negation
template <class V, tags::memory_policy MP, int ... N>
inline
tiny_array_impl<V, tags::owns_memory, N...>
operator~(tiny_array_impl<V, MP, N...> const & v)
{
    tiny_array_impl<V, tags::owns_memory, N...> res(v.size(), dont_init);
    for(int k=0; k < v.size(); ++k)
        res[k] = ~v[k];
    return res;
}

#define XTENSOR_TINYARRAY_BINARY_FUNCTION(FCT) \
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N> \
inline auto \
FCT(tiny_array_impl<V1, MP1, N...> const & l, \
    tiny_array_impl<V2, MP2, N...> const & r) \
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
template <class V, tags::memory_policy MP, class E, int ... N>
inline auto
pow(tiny_array_impl<V, MP, N...> const & v, E exponent)
{
    using namespace cmath;
    tiny_array<decltype(pow(v[0], exponent)), N...> res(v.size(), dont_init);
    for(int k=0; k < v.size(); ++k)
        res[k] = pow(v[k], exponent);
    return res;
}

    /// cross product
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int N,
          XTENSOR_REQUIRE<N == 3 || N == runtime_size> >
inline
tiny_array<promote_type_t<V1, V2>, N>
cross(tiny_array_impl<V1, MP1, N> const & r1,
      tiny_array_impl<V2, MP2, N> const & r2)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N, r1.size() == 3 && r2.size() == 3,
        "cross(): cross product requires size() == 3.");
    typedef tiny_array<promote_type_t<V1, V2>, N> Res;
    return  Res{r1[1]*r2[2] - r1[2]*r2[1],
                r1[2]*r2[0] - r1[0]*r2[2],
                r1[0]*r2[1] - r1[1]*r2[0]};
}

    /// dot product of two vectors
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int N, int M>
inline
promote_type_t<V1, V2>
dot(tiny_array_impl<V1, MP1, N> const & l,
    tiny_array_impl<V2, MP2, M> const & r)
{
    XTENSOR_ASSERT_MSG(l.size() == r.size(), "dot(): size mismatch.");
    promote_type_t<V1, V2> res = promote_type_t<V1, V2>();
    for(int k=0; k < l.size(); ++k)
        res += l[k] * r[k];
    return res;
}

    /// sum of the vector's elements
template <class V, tags::memory_policy MP, int ... N>
inline
promote_type_t<V>
sum(tiny_array_impl<V, MP, N...> const & l)
{
    promote_type_t<V> res = promote_type_t<V>();
    for(int k=0; k < l.size(); ++k)
        res += l[k];
    return res;
}

    /// mean of the vector's elements
template <class V, tags::memory_policy MP, int ... N>
inline real_promote_type_t<V>
mean(tiny_array_impl<V, MP, N...> const & t)
{
    using Promote = real_promote_type_t<V>;
    const Promote sumVal = static_cast<Promote>(sum(t));
    if(t.size() > 0)
        return sumVal / static_cast<Promote>(t.size());
    else
        return sumVal;
}

    /// cumulative sum of the vector's elements
template <class V, tags::memory_policy MP, int ... N>
inline
tiny_array_impl<promote_type_t<V>, tags::owns_memory, N...>
cumsum(tiny_array_impl<V, MP, N...> const & l)
{
    tiny_array_impl<promote_type_t<V>, tags::owns_memory, N...> res(l);
    for(int k=1; k < l.size(); ++k)
        res[k] += res[k-1];
    return res;
}

    /// product of the vector's elements
template <class V, tags::memory_policy MP, int ... N>
inline
promote_type_t<V>
prod(tiny_array_impl<V, MP, N...> const & l)
{
    using Promote = promote_type_t<V>;
    if(l.size() == 0)
        return Promote();
    Promote res = Promote(1);
    for(int k=0; k < l.size(); ++k)
        res *= l[k];
    return res;
}

    /// cumulative product of the vector's elements
template <class V, tags::memory_policy MP, int ... N>
inline
tiny_array_impl<promote_type_t<V>, tags::owns_memory, N...>
cumprod(tiny_array_impl<V, MP, N...> const & l)
{
    tiny_array_impl<promote_type_t<V>, tags::owns_memory, N...> res(l);
    for(int k=1; k < l.size(); ++k)
        res[k] *= res[k-1];
    return res;
}

    /// element-wise minimum
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
inline
tiny_array_impl<promote_type_t<V1, V2>, tags::owns_memory, N...>
min(tiny_array_impl<V1, MP1, N...> const & l,
    tiny_array_impl<V2, MP2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "min(): size mismatch.");
    tiny_array_impl<promote_type_t<V1, V2>, tags::owns_memory, N...> res(l.size(), dont_init);
    for(int k=0; k < l.size(); ++k)
        res[k] =  min(l[k], r[k]);
    return res;
}

    /// element-wise minimum with a constant
template <class V1, tags::memory_policy MP1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value>>
inline
tiny_array_impl<std::common_type_t<V1, V2>, tags::owns_memory, N...>
min(tiny_array_impl<V1, MP1, N...> const & l,
    V2 const & r)
{
    tiny_array_impl<std::common_type_t<V1, V2>, tags::owns_memory, N...> res(l.size(), dont_init);
    for(int k=0; k < l.size(); ++k)
        res[k] =  min(l[k], r);
    return res;
}

    /// element-wise minimum with a constant
template <class V1, class V2, tags::memory_policy MP2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value>>
inline
tiny_array_impl<std::common_type_t<V1, V2>, tags::owns_memory, N...>
min(V1 const & l,
    tiny_array_impl<V2, MP2, N...> const & r)
{
    tiny_array_impl<std::common_type_t<V1, V2>, tags::owns_memory, N...> res(r.size(), dont_init);
    for(int k=0; k < r.size(); ++k)
        res[k] =  min(l, r[k]);
    return res;
}

    /** Index of minimal element.

        Returns -1 for an empty array.
    */
template <class V, tags::memory_policy MP, int ... N>
inline int
min_element(tiny_array_impl<V, MP, N...> const & l)
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
template <class V, tags::memory_policy MP, int ... N>
inline
V const &
min(tiny_array_impl<V, MP, N...> const & l)
{
    int m = min_element(l);
    XTENSOR_PRECONDITION(m >= 0, "min() on empty tiny_array.");
    return l[m];
}

    /// element-wise maximum
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int ... N>
inline
tiny_array_impl<std::common_type_t<V1, V2>, tags::owns_memory, N...>
max(tiny_array_impl<V1, MP1, N...> const & l,
    tiny_array_impl<V2, MP2, N...> const & r)
{
    XTENSOR_PRECONDITION(l.size() == r.size(),
        "max(): size mismatch.");
    tiny_array_impl<std::common_type_t<V1, V2>, tags::owns_memory, N...> res(l.size(), dont_init);
    for(int k=0; k < l.size(); ++k)
        res[k] =  max(l[k], r[k]);
    return res;
}

    /// element-wise maximum with a constant
template <class V1, tags::memory_policy MP1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value>>
inline
tiny_array_impl<std::common_type_t<V1, V2>, tags::owns_memory, N...>
max(tiny_array_impl<V1, MP1, N...> const & l,
    V2 const & r)
{
    tiny_array_impl<std::common_type_t<V1, V2>, tags::owns_memory, N...> res(l.size(), dont_init);
    for(int k=0; k < l.size(); ++k)
        res[k] =  max(l[k], r);
    return res;
}

    /// element-wise maximum with a constant
template <class V1, class V2, tags::memory_policy MP2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value>>
inline
tiny_array_impl<std::common_type_t<V1, V2>, tags::owns_memory, N...>
max(V1 const & l,
    tiny_array_impl<V2, MP2, N...> const & r)
{
    tiny_array_impl<std::common_type_t<V1, V2>, tags::owns_memory, N...> res(r.size(), dont_init);
    for(int k=0; k < r.size(); ++k)
        res[k] =  max(l, r[k]);
    return res;
}

    /** Index of maximal element.

        Returns -1 for an empty array.
    */
template <class V, tags::memory_policy MP, int ... N>
inline int
max_element(tiny_array_impl<V, MP, N...> const & l)
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
template <class V, tags::memory_policy MP, int ... N>
inline V const &
max(tiny_array_impl<V, MP, N...> const & l)
{
    int m = max_element(l);
    XTENSOR_PRECONDITION(m >= 0, "max() on empty tiny_array.");
    return l[m];
}

/// squared norm
template <class V, tags::memory_policy MP, int ... N>
inline squared_norm_type_t<tiny_array_impl<V, MP, N...> >
squared_norm(tiny_array_impl<V, MP, N...> const & t)
{
    using Type = squared_norm_type_t<tiny_array_impl<V, MP, N...> >;
    Type result = Type();
    for(int i=0; i<t.size(); ++i)
        result += squared_norm(t[i]);
    return result;
}

template <class V, tags::memory_policy MP, int ... N>
inline
norm_type_t<V>
mean_square(tiny_array_impl<V, MP, N...> const & t)
{
    return norm_type_t<V>(squared_norm(t)) / t.size();
}

    /// reversed copy
template <class V, tags::memory_policy MP, int ... N>
inline
tiny_array_impl<V, tags::owns_memory, N...>
reversed(tiny_array_impl<V, MP, N...> const & t)
{
    return tiny_array_impl<V, tags::owns_memory, N...>(t.begin(), t.end(), copy_reversed);
}

    /** \brief transposed copy

        Elements are arranged such that <tt>res[k] = t[permutation[k]]</tt>.
    */
template <class V1, tags::memory_policy MP1, class V2, tags::memory_policy MP2, int N, int M>
inline
tiny_array<V1, N>
transpose(tiny_array_impl<V1, MP1, N> const & v,
          tiny_array_impl<V2, MP2, M> const & permutation)
{
    return v.transpose(permutation);
}

template <class V1, tags::memory_policy MP1, int N>
inline
tiny_array<V1, N>
transpose(tiny_array_impl<V1, MP1, N> const & v)
{
    return reversed(v);
}

template <class V1, tags::memory_policy MP1, int N1, int N2>
inline
tiny_array<V1, N2, N1>
transpose(tiny_array_impl<V1, MP1, N1, N2> const & v)
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
template <class V, tags::memory_policy MP, int ... N>
inline
tiny_array_impl<V, tags::owns_memory, N...>
clip_lower(tiny_array_impl<V, MP, N...> const & t)
{
    return clip_lower(t, V());
}

    /** \brief Clip values below a threshold.

        All elements smaller than \a val are set to \a val.
    */
template <class V, tags::memory_policy MP, int ... N>
inline
tiny_array_impl<V, tags::owns_memory, N...>
clip_lower(tiny_array_impl<V, MP, N...> const & t, const V val)
{
    tiny_array_impl<V, tags::owns_memory, N...> res(t.size(), dont_init);
    for(int k=0; k < t.size(); ++k)
    {
        res[k] = t[k] < val ? val :  t[k];
    }
    return res;
}

    /** \brief Clip values above a threshold.

        All elements bigger than \a val are set to \a val.
    */
template <class V, tags::memory_policy MP, int ... N>
inline
tiny_array_impl<V, tags::owns_memory, N...>
clip_upper(tiny_array_impl<V, MP, N...> const & t, const V val)
{
    tiny_array_impl<V, tags::owns_memory, N...> res(t.size(), dont_init);
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
template <class V, tags::memory_policy MP, int ... N>
inline
tiny_array_impl<V, tags::owns_memory, N...>
clip(tiny_array_impl<V, MP, N...> const & t,
     const V valLower, const V valUpper)
{
    tiny_array_impl<V, tags::owns_memory, N...> res(t.size(), dont_init);
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
template <class V, tags::memory_policy MP1, tags::memory_policy MP2, tags::memory_policy MP3, int ... N>
inline
tiny_array_impl<V, tags::owns_memory, N...>
clip(tiny_array_impl<V, MP1, N...> const & t,
     tiny_array_impl<V, MP2, N...> const & valLower,
     tiny_array_impl<V, MP3, N...> const & valUpper)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., t.size() == valLower.size() && t.size() == valUpper.size(),
        "clip(): size mismatch.");
    tiny_array_impl<V, tags::owns_memory, N...> res(t.size(), dont_init);
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

template <class T1, tags::memory_policy MP1, class T2, tags::memory_policy MP2, int ... N>
inline void
swap(tiny_array_impl<T1, MP1, N...> & l,
     tiny_array_impl<T2, MP2, N...> & r)
{
    l.swap(r);
}

} // namespace xt

#undef XTENSOR_ASSERT_INSIDE

#endif // XTENSOR_XTINY_HPP
