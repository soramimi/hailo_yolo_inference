#ifndef INFERENCE_H
#define INFERENCE_H

#include <cstdint>
#include <vector>

struct BBox {
	int cls;
	float x;
	float y;
	float w;
	float h;
	float score;
};

int inference(uint8_t const *input, std::vector<BBox> *bboxes);

#endif // INFERENCE_H
