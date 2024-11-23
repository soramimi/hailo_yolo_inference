#include "ImageView.h"

#include <QPainter>

ImageView::ImageView(QWidget *parent)
	: QWidget{parent}
{
}

bool ImageView::quantize(std::vector<uint8_t> *out)
{
	int w = image_.width();
	int h = image_.height();
	if (w != 640) return false;
	if (h != 640) return false;
	out->clear();
	out->resize(h * w * 3);
	for (int y = 0; y < h; y++) {
		uint8_t *dst = out->data() + 3 * w * y;
		uint8_t const *src = image_.scanLine(y);
		memcpy(dst, src, w * 3);
	}
	return true;
}

void ImageView::paintEvent(QPaintEvent *)
{
	QPainter pr(this);
	pr.drawImage(0, 0, image_);
	int img_w = image_.width();
	int img_h = image_.height();

	pr.setPen(QPen(Qt::cyan, 1));
	pr.setBrush(Qt::NoBrush);
	int n = bboxes_.size();
	for (BBox const &bb : bboxes_) {
		float x = bb.x * img_w;
		float y = bb.y * img_h;
		float w = bb.w * img_w;
		float h = bb.h * img_h;
		x -= w / 2;
		y -= h / 2;
		pr.drawRect(x, y, w, h);
	}
}
