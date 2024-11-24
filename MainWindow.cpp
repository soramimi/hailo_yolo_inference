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
		ui->centralwidget->setImage(image);
		// inference
		InferenceResult result;
		std::vector<uint8_t> vec;
		ui->centralwidget->quantize(&vec);
		inference(vec.data(), &result);
		ui->centralwidget->setInferenceResult(result);
	}
}

