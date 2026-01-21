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
__global__ void SSSP_gpu(Node *graph_nodes_av, Edge *graph_edges_av, int *cost,
    int *color, int *q1, int *q2, int *n_t,
    int *head, int *tail, int *threads_end,
    int *threads_run, int *overflow, int *gray_shade,
    int LIMIT, int CPU) {

    extern __shared__ int l_mem[];
    int* tail_bin = l_mem;
    int* l_qout = (int*)&tail_bin[1];
    int* shift = (int*)&l_qout[W_QUEUE_SIZE];
    int* base = (int*)&shift[1];

    const int tid     = threadIdx.x;
    const int gtid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int MAXWG   = gridDim.x;
    const int WG_SIZE = blockDim.x;

    // Create cuda::atomic references for system-scope synchronization (HW coherent on GH200)
    cuda::atomic_ref<int, cuda::thread_scope_system> head_ref(*head);
    cuda::atomic_ref<int, cuda::thread_scope_system> tail_ref(*tail);
    cuda::atomic_ref<int, cuda::thread_scope_system> threads_end_ref(*threads_end);
    cuda::atomic_ref<int, cuda::thread_scope_system> threads_run_ref(*threads_run);
    cuda::atomic_ref<int, cuda::thread_scope_system> n_t_ref(*n_t);
    cuda::atomic_ref<int, cuda::thread_scope_system> overflow_ref(*overflow);
    cuda::atomic_ref<int, cuda::thread_scope_system> gray_shade_ref(*gray_shade);

    int *qin, *qout;

    int iter = 1;

    int n_t_local = n_t_ref.load(cuda::memory_order_acquire);
    while(n_t_local != 0) {

        // Swap queues
        if(iter % 2 == 0) {
            qin  = q1;
            qout = q2;
        } else {
            qin  = q2;
            qout = q1;
        }

        if((n_t_local >= LIMIT) | (CPU == 0)) {

            int gray_shade_local = gray_shade_ref.load(cuda::memory_order_relaxed);

            if(tid == 0) {
                // Reset queue
                *tail_bin = 0;
            }

            // Fetch frontier elements from the queue
            if(tid == 0)
                *base = head_ref.fetch_add(WG_SIZE, cuda::memory_order_relaxed);
            __syncthreads();

            int my_base = *base;
            while(my_base < n_t_local) {

                // If local queue might overflow
                if(*tail_bin >= W_QUEUE_SIZE / 2) {
                    if(tid == 0) {
                        // Add local tail_bin to tail
                        *shift = tail_ref.fetch_add(*tail_bin, cuda::memory_order_relaxed);
                    }
                    __syncthreads();
                    int local_shift = tid;
                    while(local_shift < *tail_bin) {
                        qout[*shift + local_shift] = l_qout[local_shift];
                        // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
                        local_shift += WG_SIZE;
                    }
                    __syncthreads();
                    if(tid == 0) {
                        // Reset local queue
                        *tail_bin = 0;
                    }
                    __syncthreads();
                }

                if(my_base + tid < n_t_local && overflow_ref.load(cuda::memory_order_relaxed) == 0) {
                    // Visit a node from the current frontier
                    int pid = qin[my_base + tid];
                    //////////////// Visit node ///////////////////////////
                    cuda::atomic_ref<int, cuda::thread_scope_system> color_pid_ref(color[pid]);
                    color_pid_ref.store(BLACK, cuda::memory_order_relaxed); // Node visited
                    cuda::atomic_ref<int, cuda::thread_scope_system> cost_pid_ref(cost[pid]);
                    int  cur_cost = cost_pid_ref.load(cuda::memory_order_relaxed); // Look up shortest-path distance to this node
                    Node cur_node;
                    cur_node.x = graph_nodes_av[pid].x;
                    cur_node.y = graph_nodes_av[pid].y;
                    Edge cur_edge;
                    // For each outgoing edge
                    for(int i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
                        cur_edge.x = graph_edges_av[i].x;
                        cur_edge.y = graph_edges_av[i].y;
                        int id     = cur_edge.x;
                        int cost_local   = cur_edge.y;
                        cost_local += cur_cost;
                        cuda::atomic_ref<int, cuda::thread_scope_system> cost_id_ref(cost[id]);
                        int orig_cost = cost_id_ref.fetch_max(cost_local, cuda::memory_order_relaxed);
                        if(orig_cost < cost_local) {
                            cuda::atomic_ref<int, cuda::thread_scope_system> color_id_ref(color[id]);
                            int old_color = color_id_ref.fetch_max(gray_shade_local, cuda::memory_order_relaxed);
                            if(old_color != gray_shade_local) {
                                // Push to the queue
                                int tail_index = atomicAdd(tail_bin, 1);
                                if(tail_index >= W_QUEUE_SIZE) {
                                    overflow_ref.store(1, cuda::memory_order_relaxed);
                                    break;
                                } else
                                    l_qout[tail_index] = id;
                            }
                        }
                    }
                }
                if(tid == 0)
                    *base = head_ref.fetch_add(WG_SIZE, cuda::memory_order_relaxed); // Fetch more frontier elements from the queue
                __syncthreads();
                my_base = *base;
            }
            /////////////////////////////////////////////////////////
            // Compute size of the output and allocate space in the global queue
            if(tid == 0) {
                *shift = tail_ref.fetch_add(*tail_bin, cuda::memory_order_relaxed);
            }
            __syncthreads();
            ///////////////////// CONCATENATE INTO HOST COHERENT MEMORY /////////////////////
            int local_shift = tid;
            while(local_shift < *tail_bin) {
                qout[*shift + local_shift] = l_qout[local_shift];
                // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
                local_shift += WG_SIZE;
            }
            //////////////////////////////////////////////////////////////////////////
        }

        // Synchronization
        if(overflow_ref.load(cuda::memory_order_relaxed) == 1) {
            break;
        }

        if(CPU) { // if CPU is available
            iter++;
            if(tid == 0) {
                threads_end_ref.fetch_add(WG_SIZE, cuda::memory_order_release);

                // Spin-wait with acquire ordering for proper synchronization
                while(threads_run_ref.load(cuda::memory_order_acquire) < iter) {
                    __nanosleep(100);  // Yield to prevent busy-wait starvation
                }
                // CPU thread 0 has updated n_t, read it
                n_t_local = n_t_ref.load(cuda::memory_order_acquire);
            }
        } else { // if GPU only
            iter++;
            if(tid == 0)
                threads_end_ref.fetch_add(WG_SIZE, cuda::memory_order_release);
            if(gtid == 0) {
                // Spin-wait with acquire ordering
                while(threads_end_ref.load(cuda::memory_order_acquire) != MAXWG * WG_SIZE) {
                    __nanosleep(100);
                }
                n_t_local = tail_ref.load(cuda::memory_order_acquire);
                n_t_ref.store(n_t_local, cuda::memory_order_relaxed);
                tail_ref.store(0, cuda::memory_order_relaxed);
                head_ref.store(0, cuda::memory_order_relaxed);
                threads_end_ref.store(0, cuda::memory_order_relaxed);
                if(iter % 2 == 0)
                    gray_shade_ref.store(GRAY0, cuda::memory_order_relaxed);
                else
                    gray_shade_ref.store(GRAY1, cuda::memory_order_relaxed);
                threads_run_ref.fetch_add(1, cuda::memory_order_release);
            }
            if(tid == 0 && gtid != 0) {
                while(threads_run_ref.load(cuda::memory_order_acquire) < iter) {
                    __nanosleep(100);
                }
                n_t_local = n_t_ref.load(cuda::memory_order_acquire);
            }
        }
        __syncthreads();
        // Broadcast n_t_local to all threads in the block
        if(tid == 0) {
            *base = n_t_local;  // Reuse base as temp storage
        }
        __syncthreads();
        n_t_local = *base;
    }
}

cudaError_t call_SSSP_gpu(int blocks, int threads, Node *graph_nodes_av, Edge *graph_edges_av, int *cost,
    int *color, int *q1, int *q2, int *n_t,
    int *head, int *tail, int *threads_end, int *threads_run,
    int *overflow, int *gray_shade, int LIMIT, const int CPU, int l_mem_size){

    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    SSSP_gpu<<<dimGrid, dimBlock, l_mem_size>>>(graph_nodes_av, graph_edges_av, cost,
        color, q1, q2, n_t,
        head, tail, threads_end, threads_run,
        overflow, gray_shade, LIMIT, CPU);
    
    cudaError_t err = cudaGetLastError();
    return err;
}
