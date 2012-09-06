#pragma once

#include <thrust/memory.h>
#include <thrust/system/cuda/memory.h>

template<typename Element> class immutable_ptr;
template<typename Element> class immutable_reference;

namespace thrust
{
namespace detail
{

// XXX iterator_facade tries to instantiate the Reference
//     type when computing the answer to is_convertible<Reference,Value>
//     we can't do that at that point because immutable_reference
//     is not complete
//     WAR the problem by specializing is_convertible
template<typename T>
  struct is_convertible<immutable_reference<T>, T>
    : thrust::detail::true_type
{};

} // end detail
} // end thrust

template<typename Element>
  class immutable_ptr
    : public thrust::pointer<
               //const Element, // XXX WAR Thrust #227
               Element,
               thrust::cuda::tag,
               immutable_reference<Element>,
               immutable_ptr<Element>
             >
{
  private:
    typedef thrust::pointer<
      //const Element,
      Element, // XXX WAR Thrust #227
      thrust::cuda::tag,
      immutable_reference<Element>,
      immutable_ptr<Element>
    > super_t;

    typedef immutable_ptr<Element> pointer;

  public:
    __host__ __device__
    immutable_ptr();

    template<typename OtherElement>
    __host__ __device__
    explicit immutable_ptr(OtherElement *ptr)
    //  : super_t(ptr)
    //  XXX WAR Thrust #227
      : super_t(const_cast<Element*>(ptr))
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


template<typename T>
inline __host__ __device__
immutable_ptr<T> make_immutable(const T *ptr)
{
  return immutable_ptr<T>(ptr);
}


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
    > super_t;

  public:
    typedef immutable_ptr<Element>                               pointer;
    typedef typename thrust::detail::remove_const<Element>::type value_type;

    inline __host__ __device__
    explicit immutable_reference(const pointer &ptr)
      : super_t(ptr)
    {}

    template<typename OtherT>
    inline __host__ __device__
    immutable_reference(const immutable_reference<OtherT> &other,
                        typename thrust::detail::enable_if_convertible<
                          typename immutable_reference<OtherT>::pointer,
                          pointer
                        >::type * = 0)
      : super_t(other)
    {}

    inline __host__ __device__
    operator value_type () const
    {
    #if __CUDA_ARCH__ >= 350
      // use streaming load
      return __ldg(thrust::raw_pointer_cast(&(*this)));
    #else
      // defer to super_t otherwise
      return super_t::operator Element ();
    #endif
    };

  private:
    // can't assign to an immutable_reference
    template<typename OtherT>
    inline __host__ __device__
    immutable_reference &operator=(const immutable_reference<OtherT> &other);

    inline __host__ __device__
    immutable_reference &operator=(const value_type &x);
};

