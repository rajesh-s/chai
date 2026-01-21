/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#define _CUDA_COMPILER_

#include "support/common.h"
#include <cuda/atomic>

// CUDA kernel ------------------------------------------------------------------------------------------
__global__ void TQHistogram_gpu(task_t *queues, int *n_task_in_queue,
    int *n_written_tasks, int *n_consumed_tasks,
    int *histo, int *data, int gpuQueueSize, int frame_size, int n_bins) {

    extern __shared__ int l_mem[];
    int* last_queue = l_mem;
    task_t* t = (task_t*)&last_queue[1];
    int* l_histo = (int*)&t[1];
    
    const int tid       = threadIdx.x;
    const int tile_size = blockDim.x;

    while(true) {
        // Fetch task
        if(tid == 0) {
            int  idx_queue = *last_queue;
            int  j, jj;
            bool not_done = true;

            do {
                cuda::atomic_ref<int, cuda::thread_scope_system> consumed_ref(n_consumed_tasks[idx_queue]);
                cuda::atomic_ref<int, cuda::thread_scope_system> written_ref(n_written_tasks[idx_queue]);
                cuda::atomic_ref<int, cuda::thread_scope_system> in_queue_ref(n_task_in_queue[idx_queue]);
                
                if(consumed_ref.load(cuda::memory_order_acquire) == written_ref.load(cuda::memory_order_acquire)) {
                    idx_queue = (idx_queue + 1) % NUM_TASK_QUEUES;
                    __nanosleep(100);  // Yield to prevent busy-wait starvation
                } else {
                    if(in_queue_ref.load(cuda::memory_order_acquire) > 0) {
                        j = in_queue_ref.fetch_sub(1, cuda::memory_order_acq_rel) - 1;
                        if(j >= 0) {
                            t->id    = (queues + idx_queue * gpuQueueSize + j)->id;
                            t->op    = (queues + idx_queue * gpuQueueSize + j)->op;
                            jj       = consumed_ref.fetch_add(1, cuda::memory_order_acq_rel) + 1;
                            not_done = false;
                            if(jj == written_ref.load(cuda::memory_order_acquire)) {
                                idx_queue = (idx_queue + 1) % NUM_TASK_QUEUES;
                            }
                            *last_queue = idx_queue;
                        } else {
                            idx_queue = (idx_queue + 1) % NUM_TASK_QUEUES;
                        }
                    } else {
                        idx_queue = (idx_queue + 1) % NUM_TASK_QUEUES;
                        __nanosleep(100);
                    }
                }
            } while(not_done);
        }
        __syncthreads(); // It can be removed if work-group = wavefront

        // Compute task
        if(t->op == SIGNAL_STOP_KERNEL) {
            break;
        } else {
            if(t->op == SIGNAL_WORK_KERNEL) {
                // Reset local histogram
                for(int i = tid; i < n_bins; i += tile_size) {
                    l_histo[i] = 0;
                }
                __syncthreads();

                for(int i = tid; i < frame_size; i += tile_size) {
                    int value = (data[t->id * frame_size + i] * n_bins) >> 8;

                    atomicAdd(l_histo + value, 1);
                }

                __syncthreads();
                // Store in global memory (use atomic_ref for system-scope visibility)
                for(int i = tid; i < n_bins; i += tile_size) {
                    cuda::atomic_ref<int, cuda::thread_scope_system> histo_ref(histo[t->id * n_bins + i]);
                    histo_ref.store(l_histo[i], cuda::memory_order_release);
                }
            }
        }
    }
}

cudaError_t call_TQHistogram_gpu(int blocks, int threads, task_t *queues, int *n_task_in_queue,
    int *n_written_tasks, int *n_consumed_tasks, int *histo, int *data, int gpuQueueSize, 
    int frame_size, int n_bins, int l_mem_size){

    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    TQHistogram_gpu<<<dimGrid, dimBlock, l_mem_size>>>(queues, n_task_in_queue,
        n_written_tasks, n_consumed_tasks, histo, data, gpuQueueSize,
        frame_size, n_bins);
    
    cudaError_t err = cudaGetLastError();
    return err;
}
