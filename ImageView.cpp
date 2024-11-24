#include "ImageView.h"

#include <QPainter>

ImageView::ImageView(QWidget *parent)
	: QWidget{parent}
{
}

void ImageView::setImage(const QImage &img)
{
	image_ = img;
	result_ = {};
	update();
}

bool ImageView::quantize(std::vector<uint8_t> *out)
{
	int w = image_.width();
	int h = image_.height();
	if (w < 1 || h < 1) return false;

	w = 640;
	h = 640;

	QImage image = image_.scaled(w, h, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
	image = image.convertToFormat(QImage::Format_RGB888);

	out->clear();
	out->resize(h * w * 3);
	for (int y = 0; y < h; y++) {
		uint8_t *dst = out->data() + 3 * w * y;
		uint8_t const *src = image.scanLine(y);
		memcpy(dst, src, w * 3);
	}
	return true;
}

void ImageView::setInferenceResult(const InferenceResult &result)
{
	result_ = result;
	update();
}

void ImageView::paintEvent(QPaintEvent *)
{
	int src_w = image_.width();
	int src_h = image_.height();
	if (src_w < 1 || src_h < 1) return;

	int dw, dh;
	int dst_w = width();
	int dst_h = height();
	if (src_w * dst_h < dst_w * src_h) {
		dw = dst_h * src_w / src_h;
		dh = dst_h;
	} else {
		dw = dst_w;
		dh = dst_w * src_h / src_w;
	}
	if (dst_w < 1 || dst_h < 1) return;
	int dx = (dst_w - dw) / 2;
	int dy = (dst_h - dh) / 2;

	QPainter pr(this);
	pr.drawImage(QRect(dx, dy, dw, dh), image_, image_.rect());

	QFont font = pr.font();
	font.setPointSize(10);
	pr.setFont(font);

	pr.setOpacity(0.75);

	for (BBox const &bb : result_.bboxes) {
		float x = bb.x * dw;
		float y = bb.y * dh;
		float w = bb.w * dw;
		float h = bb.h * dh;
		x = dx + x - w / 2;
		y = dy + y - h / 2;
		pr.setPen(QPen(Qt::cyan, 1));
		pr.setBrush(Qt::NoBrush);
		pr.drawRect(x, y, w, h);
		if (bb.cls >= 0 && bb.cls < result_.labels.size()) {
			QString text = QString::fromStdString(result_.labels[bb.cls]);
			QSize sz = pr.fontMetrics().size(0, text);
			QRect r(x, y - sz.height(), sz.width(), sz.height());
			pr.fillRect(r, Qt::cyan);
			pr.setPen(Qt::black);
			pr.setBrush(Qt::NoBrush);
			pr.drawText(r, text);
		}
	}
}
