CC = icc
CXX = /opt/intel/oneapi/compiler/2023.2.0/linux/bin/intel64/icpc
LDLIBS = -lpng 
CFLAGS = -lm -O3 -march=native
hw2a: CFLAGS += -pthread
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
