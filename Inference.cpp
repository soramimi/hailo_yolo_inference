#include "Inference.h"
#include "hailo/hailort.hpp"
#include <cstring>
#include <iostream>
#include <cmath>
#include <Qt>

#if defined(__unix__)
#include <sys/mman.h>
#endif


//#define HEF_FILE ("model_3x640x640.hef")
#define HEF_FILE ("yolov9c.hef")
//#define HEF_FILE ("/home/soramimi/hailo-rpi5-examples/resources/yolov8s_h8l.hef")

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

static inline int quant(hailo_quant_info_t const &qinfo, float32_t x)
{
	float32_t y = (x / qinfo.qp_scale) + qinfo.qp_zp;
	y = roundf(y);
	return y;
}

static inline float32_t dequant(hailo_quant_info_t const &qinfo, int x)
{
	return (x - qinfo.qp_zp) * qinfo.qp_scale;
}

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

int inference(uint8_t const *input_image, std::vector<BBox> *bboxes)
{
	bboxes->clear();
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

		InferModel *model = infer_model->get();

		/* The buffers are stored here to ensure memory safety. They will only be freed once
		   the configured_infer_model is released, guaranteeing that the buffers remain intact
		   until the configured_infer_model is done using them */
		std::vector<std::shared_ptr<uint8_t>> buffer_guards;

		// Configure the infer model
		Expected<ConfiguredInferModel> configured_infer_model = model->configure();
		if (!configured_infer_model) {
			throw hailort_error(configured_infer_model.status(), "Failed to create configured infer model");
		}
		std::cout << "ConfiguredInferModel created" << std::endl;

		Expected<ConfiguredInferModel::Bindings> bindings = configured_infer_model->create_bindings();
		if (!configured_infer_model) {
			throw hailort_error(configured_infer_model.status(), "Failed to create infer bindings");
		}

		printf("Inputs:\n");
		std::vector<std::string> const &input_names = model->get_input_names();
		for (std::string const &input_name : input_names) {
			printf("  %s\n", input_name.c_str());
			size_t input_frame_size = model->input(input_name)->get_frame_size();
			std::shared_ptr<uint8_t> input_buffer = page_aligned_alloc(input_frame_size);
			hailo_status status = bindings->input(input_name)->set_buffer(MemoryView(input_buffer.get(), input_frame_size));
			if (HAILO_SUCCESS != status) {
				throw hailort_error(status, "Failed to set infer input buffer");
			}

			buffer_guards.push_back(input_buffer);
		}

		printf("Outputs:\n");
		std::vector<std::string> const &output_names = model->get_output_names();
		for (std::string const &output_name : output_names) {
			printf("  %s\n", output_name.c_str());
			size_t output_frame_size = model->output(output_name)->get_frame_size();
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
				memcpy(idata, input_image, isize);
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

#if 0
		std::vector<hailo_quant_info_t> iqinfo = model->input()->get_quant_infos();
		std::vector<hailo_quant_info_t> oqinfo = model->output()->get_quant_infos();
		int y1 = quant(iqinfo[0], 0.3333);
		float y2 = dequant(iqinfo[0], 0xaa);

		hailo_3d_image_shape_t shape = model->output()->shape();
#endif

		{
			Expected<MemoryView> in = bindings->input()->get_buffer();
			Expected<MemoryView> out = bindings->output()->get_buffer();
			if (in && out) {
				uint8_t const *idata = in->data();
				uint8_t const *odata = out->data();

				Expected<hailo_nms_shape_t> result = model->output()->get_nms_shape();

				size_t isize = in->size();
				Q_ASSERT(isize == 640 * 640 * 3);

				size_t osize = out->size();
				Q_ASSERT(osize == (sizeof(float) + sizeof(hailo_bbox_float32_t) * result->max_bboxes_per_class) * result->number_of_classes);

				uint8_t const *ptr = odata;
				int ncls = result->number_of_classes;
				for (int cls = 0; cls < ncls; cls++) {
					int nbox = *(float const *)ptr;
					ptr += sizeof(float);
					hailo_bbox_float32_t const *bbox = (hailo_bbox_float32_t const *)ptr;
					for (int box = 0; box < nbox; box++) {
						BBox bb;
						bb.cls = cls;
						bb.x = (bbox->x_min + bbox->x_max) / 2;
						bb.y = (bbox->y_min + bbox->y_max) / 2;
						bb.w = bbox->x_max - bbox->x_min;
						bb.h = bbox->y_max - bbox->y_min;
						bb.score = bbox->score;
						bboxes->push_back(bb);
						bbox++;
					}
					ptr += sizeof(hailo_bbox_float32_t) * nbox;
				}
			}
		}

		std::cout << "Inference finished successfully" << std::endl;
	} catch (const hailort_error &exception) {
		std::cout << "Failed to run inference. status=" << exception.status() << ", error message: " << exception.what() << std::endl;
		return exception.status();
	};

	return HAILO_SUCCESS;
}


