#include "Inference.h"
#include "MainWindow.h"
#include "ui_MainWindow.h"
#include <QFileDialog>

#include <cstdint>
#include <vector>

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	ui->setupUi(this);
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::on_action_file_open_triggered()
{
	QString path;
	path = QFileDialog::getOpenFileName(this, tr("Open image file"), path);
	if (!path.isEmpty()) {
		// set image
		QImage image;
		image.load(path);
		image = image.scaled(640, 640, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
		image = image.convertToFormat(QImage::Format_RGB888);
		ui->centralwidget->setImage(image);
		// inference
		std::vector<BBox> bboxes;
		std::vector<uint8_t> vec;
		ui->centralwidget->quantize(&vec);
		inference(vec.data(), &bboxes);
		ui->centralwidget->setBoundingBoxes(std::move(bboxes));
	}
}

