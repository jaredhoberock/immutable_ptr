#include "immutable_ptr.hpp"

template<typename Element>
  Element immutable_reference<Element>
    ::operator Element () const
{
#ifdef __CUDA_ARCH__
  // use streaming load
  return __ldg(thrust::raw_pointer_cast(&(*this)));
#else
  // defer to super_t on the host
  return super_t::operator Element ();
#endif
};

