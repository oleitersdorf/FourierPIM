CC=/usr/local/cuda-11.7/bin/nvcc
CFLAGS=-Icommon -lcufft -arch=native
HALF = half
FULL = full

ifeq ($(DTYPE), $(HALF))
$(eval CFLAGS += -DHALF_PRECISION)
else # default = FULL
$(eval CFLAGS += -DFULL_PRECISION)
endif

all: clean fft complex_poly real_poly

fft: fft.cu
	$(CC) -o $@ $< $(CFLAGS)

complex_poly: complex_poly.cu
	$(CC) -o $@ $< $(CFLAGS)

real_poly: real_poly.cu
	$(CC) -o $@ $< $(CFLAGS)

clean:
	rm -f fft complex_poly real_poly

