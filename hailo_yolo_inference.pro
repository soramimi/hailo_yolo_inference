DESTDIR = $$PWD
TARGET = hailo_yolo_inference
TEMPLATE = app
QT = core gui widgets

LIBS += -lhailort

SOURCES += \
	ImageView.cpp \
	Inference.cpp \
	MainWindow.cpp \
	main.cpp

FORMS += \
	MainWindow.ui

HEADERS += \
	ImageView.h \
	Inference.h \
	MainWindow.h

