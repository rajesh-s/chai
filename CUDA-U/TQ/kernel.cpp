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

#include "kernel.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <cuda/atomic>
#include <mutex>

// Type alias for system-scope atomic int
using atomic_int_sys = cuda::atomic<int, cuda::thread_scope_system>;

// Mutex for protecting shared state between CPU threads
static std::mutex g_insert_mutex;

//----------------------------------------------------------------------------
// CPU: Host enqueue task (thread-safe version)
//----------------------------------------------------------------------------
void host_insert_tasks(task_t *queues, task_t *task_pool, atomic_int_sys *n_consumed_tasks,
    atomic_int_sys *n_written_tasks, atomic_int_sys *n_task_in_queue, atomic_int_sys *last_queue, 
    int n_tasks_to_insert, int gpu_queue_size, int pool_offset) {
    
    int i = last_queue->load(cuda::memory_order_acquire);
    int n_total_tasks         = n_tasks_to_insert;
    int n_remaining_tasks     = n_tasks_to_insert;
    int n_tasks_to_write_next = (n_remaining_tasks > gpu_queue_size) ? gpu_queue_size : n_remaining_tasks;
    int n_tasks_this_batch    = n_tasks_to_write_next;
#if PRINT
    printf("Inserting Tasks (%d to insert)\t \n", n_tasks_to_write_next);
#endif
    do {
#if PRINT
        printf("Loop iteration... i = %d, consumed=%d, written=%d\n", i, n_consumed_tasks[i].load(cuda::memory_order_acquire),
            n_written_tasks[i].load(cuda::memory_order_acquire));
#endif
        if(n_consumed_tasks[i].load(cuda::memory_order_acquire) == n_written_tasks[i].load(cuda::memory_order_acquire)) {
#if PRINT
            printf("Inserting Tasks... %d (%d) in queue %d\n", n_remaining_tasks, n_tasks_this_batch, i);
#endif
            // Insert tasks in queue i
            memcpy(&queues[i * gpu_queue_size], &task_pool[pool_offset + n_total_tasks - n_remaining_tasks],
                n_tasks_this_batch * sizeof(task_t));
            // Memory fence to ensure memcpy is visible before updating atomics
            cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
            // Update number of tasks in queue i
            n_task_in_queue[i].store(n_tasks_this_batch, cuda::memory_order_release);
            // Total number of tasks written in queue i
            n_written_tasks[i].fetch_add(n_tasks_this_batch, cuda::memory_order_release);
            // Next queue
            i = (i + 1) % NUM_TASK_QUEUES;
            // Remaining tasks
            n_remaining_tasks -= n_tasks_to_write_next;
            n_tasks_to_write_next = (n_remaining_tasks > gpu_queue_size) ? gpu_queue_size : n_remaining_tasks;
            n_tasks_this_batch    = n_tasks_to_write_next;
        } else {
            i = (i + 1) % NUM_TASK_QUEUES;
        }
    } while(n_tasks_to_write_next > 0);

    last_queue->store(i, cuda::memory_order_release);
}

void run_cpu_threads(int n_threads, task_t *queues, atomic_int_sys *n_task_in_queue,
    atomic_int_sys *n_written_tasks, atomic_int_sys *n_consumed_tasks, task_t *task_pool,
    int *data, int gpu_queue_size, atomic_int_sys *offset, atomic_int_sys *last_queue, 
    int tpi, int poolSize, int n_work_groups) {
///////////////// Run CPU worker threads /////////////////////////////////
#if PRINT
    printf("Starting %d CPU threads\n", n_threads);
#endif

    // Atomic flag to track if stop tasks have been sent
    atomic_int_sys stop_sent(0);

    std::vector<std::thread> cpu_threads;
    for(int i = 0; i < n_threads; i++) {

        cpu_threads.push_back(std::thread([&, i]() {

            int maxConcurrentBlocks = n_work_groups;

            while(true) {
                int my_offset;
                int tasks_to_insert;
                bool should_stop = false;
                
                // Critical section: atomically claim a batch of tasks
                {
                    std::lock_guard<std::mutex> lock(g_insert_mutex);
                    my_offset = offset->load(cuda::memory_order_acquire);
                    
                    if(my_offset >= poolSize) {
                        // No more regular tasks - check if we need to send stop tasks
                        if(stop_sent.load(cuda::memory_order_acquire) == 0) {
                            // First thread to reach here sends stop tasks
                            stop_sent.store(1, cuda::memory_order_release);
                            
                            // Create stop tasks
                            for(int j = 0; j < maxConcurrentBlocks; j++) {
                                (task_pool + j)->id = -1;
                                (task_pool + j)->op = SIGNAL_STOP_KERNEL;
                            }
                            cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
                            
                            // Insert stop tasks
                            host_insert_tasks(queues, task_pool, n_consumed_tasks, n_written_tasks,
                                n_task_in_queue, last_queue, maxConcurrentBlocks, gpu_queue_size, 0);
#if PRINT
                            printf("Thread %d sent STOP tasks\n", i);
#endif
                        }
                        should_stop = true;
                    } else {
                        // Claim tasks
                        tasks_to_insert = (poolSize - my_offset >= tpi) ? tpi : (poolSize - my_offset);
                        offset->fetch_add(tasks_to_insert, cuda::memory_order_acq_rel);
                    }
                }
                
                if(should_stop) {
                    break;
                }
                
                // Insert tasks outside the critical section (but host_insert_tasks 
                // still needs to be serialized due to queue state)
                {
                    std::lock_guard<std::mutex> lock(g_insert_mutex);
                    host_insert_tasks(queues, task_pool, n_consumed_tasks, n_written_tasks,
                        n_task_in_queue, last_queue, tasks_to_insert, gpu_queue_size, my_offset);
                }
                
#if PRINT
                printf("Thread %d inserted %d tasks at offset %d\n", i, tasks_to_insert, my_offset);
#endif
            }

        }));
    }

    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
