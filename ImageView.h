#ifndef IMAGEVIEW_H
#define IMAGEVIEW_H

#include "Inference.h"

#include <QWidget>

class ImageView : public QWidget {
	Q_OBJECT

private:
	QImage image_;
	InferenceResult result_;
protected:
	void paintEvent(QPaintEvent *event);
public:
	explicit ImageView(QWidget *parent = nullptr);
	void setImage(QImage const &img);
	bool quantize(std::vector<uint8_t> *out);
	void setInferenceResult(InferenceResult const &result);
signals:


};

#endif // IMAGEVIEW_H
