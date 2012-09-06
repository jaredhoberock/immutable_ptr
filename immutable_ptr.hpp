#pragma once

#include <thrust/memory.h>
#include <thrust/system/cuda/memory.h>

template<typename Element> class immutable_reference;

template<typename Element>
  class immutable_ptr
    : public thrust::pointer<const Element,
                             thrust::cuda::tag,
                             immutable_reference<Element>,
                             immutable_ptr<Element>
{
  private:
    typedef thrust::pointer<
      Element,
      thrust::cuda::tag,
      immutable_reference<T>,
      immutable_ptr<Element>
    > super_t;

  public:
    __host__ __device__
    immutable_ptr();

    template<typename OtherElement>
    __host__ __device__
    explicit immutable_ptr(OtherElement *ptr)
      : super_t(ptr)
    {}

    template<typename OtherPointer>
    __host__ __device__
    immutable_ptr(const OtherPointer &other,
                  typename thrust::detail::enable_if_pointer_is_convertible<
                    OtherPointer,
                    immutable_ptr<Element>
                  >::type * = 0)
      : super_t(other)
    {}

    template<typename OtherPointer>
    __host__ __device__
    typename thrust::detail::enable_if_pointer_is_convertible<
      OtherPointer,
      pointer,
      immutable_ptr &
    >::type
    operator=(const OtherPointer &other)
    {
      return super_t::operator=(other);
    }
};

template<typename Element>
  class immutable_reference
    : public thrust::reference<
               const Element,
               immutable_ptr<Element>,
               immutable_reference<Element>
             >
{
  private:
    typedef thrust::reference<
      const Element,
      immutable_ptr<Element>,
      immutable_reference<Element>
    > type;

  public:
    typedef Pointer                                              pointer;
    typedef typename thrust::detail::remove_const<Element>::type value_type;

    __host__ __device__
    explicit immutable_reference(const pointer &ptr)
      : super_t(ptr)
    {}

    template<typename OtherT>
    __host__ __device__
    immutable_reference(const immutable_reference<OtherT> &other,
                        typename thrust::detail::enable_if_convertible<
                          typename reference<OtherT>::pointer,
                          pointer
                        >::type * = 0)
      : super_t(other)
    {}

    __host__ __device__
    operator value_type () const;

  private:
    // can't assign to an immutable_reference
    template<typename OtherT>
    __host__ __device__
    reference &operator=(const reference<OtherT> &other);

    __host__ __device__
    reference &operator=(const value_type &x);
};

#include "immutable_ptr.hpp"

