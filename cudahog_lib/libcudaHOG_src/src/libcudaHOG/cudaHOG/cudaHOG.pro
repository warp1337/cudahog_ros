TEMPLATE = lib
CONFIG += debug console

CXXFLAGS += -fPIC
CFLAGS += -fPIC

HEADERS = cudaHOG.h gradients.h padding.h conversions.h blocks.h global.h detections.h parameters.h
SOURCES = cudaHOG.cpp parameters.cpp
CUSOURCES = gradients.cu padding.cu conversions.cu blocks.cu hog.cu descriptor.cu \
			svm.cu timer.cu detections.cu nms.cu

LIBS += -L/usr/local/cuda/lib64 -lcudart

QMAKE_CUC = nvcc
cu.name = Cuda ${QMAKE_FILE_IN}
cu.input = CUSOURCES
cu.CONFIG += no_link
cu.variable_out = OBJECTS

INCLUDEPATH += $(CUDA_INC_PATH)
QMAKE_CUFLAGS += $$QMAKE_CFLAGS
## QMAKE_CUEXTRAFLAGS += -arch=sm_20 --ptxas-options=-v -Xcompiler -fPIC -Xcompiler $$join(QMAKE_CUFLAGS, ",")
QMAKE_CUEXTRAFLAGS += -arch=sm_20 -Xcompiler -fPIC -Xcompiler $$join(QMAKE_CUFLAGS, ",")
QMAKE_CUEXTRAFLAGS += $(DEFINES) $(INCPATH) $$join(QMAKE_COMPILER_DEFINES, " -D", -D)
QMAKE_CUEXTRAFLAGS += -c

cu.commands = $$QMAKE_CUC $$QMAKE_CUEXTRAFLAGS -o ${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ} ${QMAKE_FILE_NAME}$$escape_expand(\n\t)
cu.output = ${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ}
silent:cu.commands = @echo nvcc ${QMAKE_FILE_IN} && $$cu.commands
QMAKE_EXTRA_COMPILERS += cu

build_pass|isEmpty(BUILDS):cuclean.depends = compiler_cu_clean
else:cuclean.CONFIG += recursive
QMAKE_EXTRA_TARGETS += cuclean
