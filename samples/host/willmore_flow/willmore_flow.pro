include($${PWD}/../../../common.pri)

TEMPLATE = app
TARGET = willmore_flow.out

OBJECTS_DIR = obj

SOURCES += $$files(src/main_v2.cpp, true)

INCLUDEPATH += $$PWD/include 


INCLUDEPATH +=   /usr/include/
LIBS        += -L/usr/include/
LIBS        += -L/usr/local/include/
LIBS        += -L/usr/local/include/CGAL
LIBS        += -lgmp

LIBS += -L../../../flo/lib
# -lflo 
QMAKE_RPATHDIR += ../../../flo/lib


