/**
 * Copyright (c) 2020-2023 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file async_infer_basic_example.cpp
 * This example demonstrates the Async Infer API usage with a specific model.
 **/

#include "hailo/hailort.hpp"

#include <cstring>
#include <iostream>

#if defined(__unix__)
#include <sys/mman.h>
#endif

#define HEF_FILE ("model_3x640x640.hef")

using namespace hailort;

class hailort_error : public std::runtime_error {
private:
	hailo_status status_;
public:
	hailort_error(hailo_status status, std::string const &what)
		: runtime_error(what)
		, status_(status)
	{
	}
	int status() const
	{
		return status_;
	}
};


static std::shared_ptr<uint8_t> page_aligned_alloc(size_t size)
{
#if defined(__unix__)
	void *addr = mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	if (MAP_FAILED == addr) throw std::bad_alloc();
	return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [size](void *addr) { munmap(addr, size); });
#elif defined(_MSC_VER)
	void *addr = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
	if (!addr) throw std::bad_alloc();
	return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [](void *addr){ VirtualFree(addr, 0, MEM_RELEASE); });
#else
#pragma error("Aligned alloc not supported")
#endif
}

int main()
{
	try {
		Expected<std::unique_ptr<VDevice>> vdevice = VDevice::create();
		if (!vdevice) {
			throw hailort_error(vdevice.status(), "Failed crete vdevice");
		}
		std::cout << "VDevice created" << std::endl;

		// Create infer model from HEF file.
		Expected<std::shared_ptr<InferModel>> infer_model = (*vdevice)->create_infer_model(HEF_FILE);
		if (!infer_model) {
			throw hailort_error(infer_model.status(), "Failed to create infer model");
		}
		std::cout << "InferModel created" << std::endl;

		/* The buffers are stored here to ensure memory safety. They will only be freed once
		   the configured_infer_model is released, guaranteeing that the buffers remain intact
		   until the configured_infer_model is done using them */
		std::vector<std::shared_ptr<uint8_t>> buffer_guards;

		// Configure the infer model
		Expected<ConfiguredInferModel> configured_infer_model = (*infer_model)->configure();
		if (!configured_infer_model) {
			throw hailort_error(configured_infer_model.status(), "Failed to create configured infer model");
		}
		std::cout << "ConfiguredInferModel created" << std::endl;

		Expected<ConfiguredInferModel::Bindings> bindings = configured_infer_model->create_bindings();
		if (!configured_infer_model) {
			throw hailort_error(configured_infer_model.status(), "Failed to create infer bindings");
		}

		printf("Inputs:\n");
		std::vector<std::string> const &input_names = (*infer_model)->get_input_names();
		for (std::string const &input_name : input_names) {
			printf("  %s\n", input_name.c_str());
			size_t input_frame_size = (*infer_model)->input(input_name)->get_frame_size();
			std::shared_ptr<uint8_t> input_buffer = page_aligned_alloc(input_frame_size);
			hailo_status status = bindings->input(input_name)->set_buffer(MemoryView(input_buffer.get(), input_frame_size));
			if (HAILO_SUCCESS != status) {
				throw hailort_error(status, "Failed to set infer input buffer");
			}

			buffer_guards.push_back(input_buffer);
		}

		printf("Outputs:\n");
		std::vector<std::string> const &output_names = (*infer_model)->get_output_names();
		for (std::string const &output_name : output_names) {
			printf("  %s\n", output_name.c_str());
			size_t output_frame_size = (*infer_model)->output(output_name)->get_frame_size();
			std::shared_ptr<uint8_t> output_buffer = page_aligned_alloc(output_frame_size);
			hailo_status status = bindings->output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
			if (HAILO_SUCCESS != status) {
				throw hailort_error(status, "Failed to set infer output buffer");
			}

			buffer_guards.push_back(output_buffer);
		}
		std::cout << "ConfiguredInferModel::Bindings created and configured" << std::endl;

		// Fill input buffer
		{
			Expected<MemoryView> in = bindings->input()->get_buffer();
			if (in) {
				uint8_t *idata = in->data();
				size_t isize = in->size();
				memset(idata, 0x55, isize);
			}
		}

		std::cout << "Running inference..." << std::endl;
		// Run the async infer job
		Expected<AsyncInferJob> job = configured_infer_model->run_async(*bindings);
		if (!job) {
			throw hailort_error(job.status(), "Failed to start async infer job");
		}
		hailo_status status = job->wait(std::chrono::milliseconds(1000));
		if (HAILO_SUCCESS != status) {
			throw hailort_error(status, "Failed to wait for infer to finish");
		}

		// Print result
		{
			Expected<MemoryView> in = bindings->input()->get_buffer();
			Expected<MemoryView> out = bindings->output()->get_buffer();
			if (in && out) {
				uint8_t const *idata = in->data();
				size_t isize = in->size();
				uint8_t const *odata = out->data();
				size_t osize = out->size();
				printf(" input: size=%u [0]=%02x\n", isize, *idata);
				printf("output: size=%u [0]=%02x\n", osize, *odata);
			}
		}

		std::cout << "Inference finished successfully" << std::endl;
	} catch (const hailort_error &exception) {
		std::cout << "Failed to run inference. status=" << exception.status() << ", error message: " << exception.what() << std::endl;
		return exception.status();
	};

	return HAILO_SUCCESS;
}
