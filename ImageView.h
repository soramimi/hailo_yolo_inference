#ifndef IMAGEVIEW_H
#define IMAGEVIEW_H

#include "Inference.h"

#include <QWidget>

class ImageView : public QWidget {
	Q_OBJECT

private:
	QImage image_;
	std::vector<BBox> bboxes_;
protected:
	void paintEvent(QPaintEvent *event);
public:
	explicit ImageView(QWidget *parent = nullptr);
	void setImage(QImage const &img)
	{
		image_ = img;
		bboxes_ = {};
		update();
	}
	bool quantize(std::vector<uint8_t> *out);
	void setBoundingBoxes(std::vector<BBox> &&bb)
	{
		bboxes_ = std::move(bb);
		update();
	}
signals:


};

#endif // IMAGEVIEW_H
