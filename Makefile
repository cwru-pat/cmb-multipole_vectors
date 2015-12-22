# Build C mpd_driver

CC=gcc
OPTIONS=-Wall 
DEBUG=-g -O2
CFLAGS+=
DEFINES+=
INCLUDES+=
CFLAGS+=$(INCLUDES) $(DEBUG) $(OPTIONS) $(DEFINES)

# Extra shorthand for me
GSLLIBS= -lgsl -lgslcblas

MPD_DRIVER_OBJS=mpd_driver.o mpd_decomp.o

mpd_driver: $(MPD_DRIVER_OBJS)
	$(CC) $(CFLAGS) -o $@ $(MPD_DRIVER_OBJS) $(GSLLIBS) -lm

clean: 
	rm -f *.o *.out *~ core
