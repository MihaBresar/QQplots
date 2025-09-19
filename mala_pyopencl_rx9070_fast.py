# mala_pyopencl_rx9070_fix.py
# AMD RX 9070 (gfx1201) OpenCL MALA — robust launch + no NumPy overflow warnings.

import time, numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

# ======== TUNABLES ========
N_TOTAL          = 30_000_000   # steps per chain
BURNIN           = 100
N_CHAINS         = 200_000      # try 100k–300k for better utilization
STEP_SIZE        = 0.55
STEPS_PER_LAUNCH = 6000         # inner-iter does 2 steps -> 12k steps/launch
DESIRED_LOCAL    = 256          # we'll auto-clamp to valid size
SEED             = 123456789

# Force RX 9070 from your listing: Platform 0, Device 1
PLATFORM_INDEX   = 0
DEVICE_INDEX     = 1

# ======== KERNEL (2 MALA steps per inner loop) ========
KERNEL = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline uint xorshift32(uint x){ x^=x<<13; x^=x>>17; x^=x<<5; return x; }
inline float u01(uint *s){
    uint x = xorshift32(*s); *s = x;
    return fmax((x + 1u) * 2.3283064365386963e-10f, 1.0e-7f);
}
inline void randn_pair(uint *s, float *z0, float *z1){
    float u1 = u01(s), u2 = u01(s);
    float r = sqrt(-2.0f * log(u1));
    float a = 6.283185307179586f * u2; // 2*pi
    *z0 = r * cos(a); *z1 = r * sin(a);
}
inline float log_unnorm_t4(float x){
    float x2=x*x, x4=x2*x2;
    return -log(1.0f + 0.25f*x4);
}
inline float grad_log_unnorm_t4(float x){
    float x2=x*x, x4=x2*x2;
    return -x / (2.0f*(1.0f + 0.25f*x4));
}

// Added 'n' and early return so we can pad global size safely
__kernel void mala_chunk2(
    __global float  *x, __global float  *logp,
    __global double *sum_abs, __global double *sum_ind,
    __global int    *burn_left, __global uint *rng_state,
    const int n,
    const float step, const float ss2, const float inv2s, const int K)
{
    int i = get_global_id(0);
    if (i >= n) return;

    float xx = x[i];
    float lp = logp[i];
    int   b  = burn_left[i];
    uint  s  = rng_state[i];

    for (int it=0; it<K; ++it){
