/*
    Usage: ./mem_test.out [global size] [local size] [num iters]
*/

#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>

using Type = float;
constexpr int DEF_GLOBAL_SIZE = 1024 * 1024 * 8;
constexpr int DEF_LOCAL_ZIE = 512;
constexpr int DEF_NUM_ITERS = 1;

int main(int argc, char** argv) {

    // Retrive Input
    size_t global_size = argc >= 2 ? atoi(argv[1]) : DEF_GLOBAL_SIZE;
    size_t local_size = argc >= 3 ? atoi(argv[2]) : DEF_LOCAL_ZIE;
    size_t num_iters = argc >= 4 ? atoi(argv[3]) : DEF_NUM_ITERS;

    // Create SYCL queue and device selector
    sycl::queue myQueue(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});

    auto vecH = sycl::malloc_host<Type>(global_size, myQueue);
    auto vecA = sycl::malloc_device<Type>(global_size, myQueue);
    auto vecB = sycl::malloc_device<Type>(global_size, myQueue);
    auto vecC = sycl::malloc_device<Type>(global_size, myQueue);

    // Initialize input data
    {
        for (size_t i = 0; i < global_size; i++) {
            vecH[i] = static_cast<Type>(i);
        }
        myQueue.copy<Type>(vecH, vecA, global_size);
        myQueue.copy<Type>(vecH, vecB, global_size);
        myQueue.fill<Type>(vecC, static_cast<Type>(0), global_size);
        myQueue.wait();
    }

    // Submit the SYCL kernel
    auto e = myQueue.submit([&](sycl::handler& cgh) {
        // Define a local memory accessor
        auto localAcc = sycl::local_accessor<Type, 1>(sycl::range<1>(local_size), cgh);

        cgh.parallel_for<class vector_add>(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            size_t globalIdx = item.get_global_id(0);
            size_t localIdx = item.get_local_id(0);

            // Iterate to increase the GPU time
            for (int k = 0; k < num_iters; k++) {
                // Load data from global memory to local memory
                localAcc[localIdx] = vecA[globalIdx] + vecB[globalIdx];
                item.barrier(sycl::access::fence_space::local_space);

                // Store data from local memory to global memory
                vecC[globalIdx] = localAcc[localIdx];
            }
        });
    });
    myQueue.wait_and_throw();

    // Measure execution time
    auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    double executionTime = (end - start);  // Convert to milliseconds

    // Calculate memory bandwidth
    size_t dataSizeBytes = global_size * sizeof(Type);
    double bandwidthGBs = dataSizeBytes / executionTime;

    // Print results
    std::cout << "----------------------------" << std::endl;
    std::cout << "Global Size: " << global_size << std::endl;
    std::cout << "Local Size: " << local_size << std::endl;
    std::cout << "Num Iterations: " << num_iters << std::endl;
    std::cout << "Execution Time: " << (executionTime / 1e6) << " ms\n";
    std::cout << "Memory Bandwidth: " << bandwidthGBs << " GB/s\n";

    // Verify the result (optional)
    {
        myQueue.copy(vecC, vecH, global_size);
        myQueue.wait();

        for (size_t i = 0; i < global_size; i++) {
            Type expected = 2.0f * i;  // The result should be twice the value of each input element
            if (vecH[i] != expected) {
                std::cout << "Error: Mismatch at index " << i << ". Expected: " << expected << ", Actual: " << vecH[i] << std::endl;
                break;
            }
        }
    }

    sycl::free(vecA, myQueue);
    sycl::free(vecB, myQueue);
    sycl::free(vecC, myQueue);

    return 0;
}
