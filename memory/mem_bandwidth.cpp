#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>

using Type = float;

int main(int argc, char** argv) {

    // Retrive Input
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <global_size> <local_size>" << std::endl;
        return 1;
    }
    size_t global_size = atoi(argv[1]);
    size_t local_size = atoi(argv[2]);

    // Create SYCL queue and device selector
    sycl::queue myQueue(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});

    auto vecA = sycl::malloc_shared<Type>(global_size, myQueue);
    auto vecB = sycl::malloc_shared<Type>(global_size, myQueue);
    auto vecC = sycl::malloc_shared<Type>(global_size, myQueue);

    // Initialize input data
    {
        for (size_t i = 0; i < global_size; i++) {
            vecA[i] = static_cast<Type>(i);
            vecB[i] = static_cast<Type>(i);
        }
    }

    // Submit the SYCL kernel
    auto e = myQueue.submit([&](sycl::handler& cgh) {
        // Define a local memory accessor
        auto localAcc = sycl::local_accessor<Type, 1>(sycl::range<1>(local_size), cgh);

        cgh.parallel_for<class vector_add>(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            size_t globalIdx = item.get_global_id(0);
            size_t localIdx = item.get_local_id(0);

            // Load data from global memory to local memory
            localAcc[localIdx] = vecA[globalIdx] + vecB[globalIdx];
            item.barrier(sycl::access::fence_space::local_space);

            // Store data from local memory to global memory
            vecC[globalIdx] = localAcc[localIdx];
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
    std::cout << "Execution Time: " << (executionTime / 1e6) << " ms\n";
    std::cout << "Memory Bandwidth: " << bandwidthGBs << " GB/s\n";

    // Verify the result (optional)
    {
        for (size_t i = 0; i < global_size; i++) {
            Type expected = 2.0f * i;  // The result should be twice the value of each input element
            if (vecC[i] != expected) {
                std::cout << "Error: Mismatch at index " << i << ". Expected: " << expected << ", Actual: " << vecC[i] << std::endl;
                break;
            }
        }
    }

    sycl::free(vecA, myQueue);
    sycl::free(vecB, myQueue);
    sycl::free(vecC, myQueue);

    return 0;
}
