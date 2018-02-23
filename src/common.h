

#ifndef COMMON_H
#define COMMON_H

#include "stdlib.h"
#include "string.h"

#define RNN_INLINE inline
#define OPUS_INLINE inline


/** RNNoise wrapper for malloc(). To do your own dynamic allocation, all you need t
o do is replace this function and rnnoise_free */
#ifndef OVERRIDE_RNNOISE_ALLOC
static RNN_INLINE void *rnnoise_alloc (size_t size)
{
   return malloc(size);
}
#endif

/** RNNoise wrapper for free(). To do your own dynamic allocation, all you need to do is replace this function and rnnoise_alloc */
#ifndef OVERRIDE_RNNOISE_FREE
static RNN_INLINE void rnnoise_free (void *ptr)
{
   free(ptr);
}
#endif

/** Copy n elements from src to dst. The 0* term provides compile-time type checking  */
#ifndef OVERRIDE_RNN_COPY
#define RNN_COPY(dst, src, n) (memcpy((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Copy n elements from src to dst, allowing overlapping regions. The 0* term
    provides compile-time type checking */
#ifndef OVERRIDE_RNN_MOVE
#define RNN_MOVE(dst, src, n) (memmove((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Set n elements of dst to zero */
#ifndef OVERRIDE_RNN_CLEAR
#define RNN_CLEAR(dst, n) (memset((dst), 0, (n)*sizeof(*(dst))))
#endif

#define PRINT_FIRST_5(src) (fprintf(stderr, "%f, %f, %f, %f, %f \n", src[0], src[1], src[2], src[3], src[4]))
#define PRINT_FIRST_5_r(src) (fprintf(stderr, "%f, %f, %f, %f, %f \n", src[0].r, src[1].r, src[2].r, src[3].r, src[4].r))
#define PRINT_FIRST_5_i(src) (fprintf(stderr, "%f, %f, %f, %f, %f \n", src[0].i, src[1].i, src[2].i, src[3].i, src[4].i))

#define PRINT_FIRST_4(src) (fprintf(stderr, "%f, %f, %f, %f \n", src[0], src[1], src[2], src[3]))

#endif
