#ifndef INFERENCE_H
#define INFERENCE_H

#include <cstdint>
#include <vector>
#include <string>

struct BBox {
	int cls;
	float x;
	float y;
	float w;
	float h;
	float score;
};

class InferenceResult {
public:
	std::vector<std::string> labels;
	std::vector<BBox> bboxes;
};

int inference(uint8_t const *input, InferenceResult *result);

#endif // INFERENCE_H
