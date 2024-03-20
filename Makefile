CC = gcc

CFLAGS = -Wall -Wextra -D CL_TARGET_OPENCL_VERSION=300

SRC = mult.c

EXECUTABLE = mult

LIBS = -lOpenCL

all: $(EXECUTABLE)

$(EXECUTABLE): $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

clean:
	rm -f $(EXECUTABLE)
