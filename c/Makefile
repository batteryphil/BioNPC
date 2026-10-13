# BioNPC C / Cosmopolitan APE Engine Makefile
#
# Builds high-performance, 100% memory-safe Actually Portable Executables
# using the Fil-C Cosmopolitan toolchain.

FILC_CLANG ?= /home/phil/.gemini/antigravity/scratch/fil-c/build/bin/clang
CC := $(FILC_CLANG)

CFLAGS ?= -O3 -Wall -Wextra -Iinclude
LDFLAGS ?= -lm

SRCDIR := src
INCDIR := include
EXDIR := examples
BINDIR := bin
LIBDIR := lib

OBJS := $(SRCDIR)/bionpc.o

BENCH_BIN := $(BINDIR)/bionpc_bench.com
SIM_BIN := $(BINDIR)/wolf_sheep_sim.com
TEST_BIN := $(BINDIR)/test_io.com
STATIC_LIB := $(LIBDIR)/libbionpc.a

.PHONY: all clean test bench sim native

all: $(BENCH_BIN) $(SIM_BIN) $(TEST_BIN) $(STATIC_LIB)

$(BINDIR):
	mkdir -p $(BINDIR)

$(LIBDIR):
	mkdir -p $(LIBDIR)

$(SRCDIR)/bionpc.o: $(SRCDIR)/bionpc.c $(INCDIR)/bionpc.h
	$(CC) $(CFLAGS) -c $< -o $@

$(STATIC_LIB): $(SRCDIR)/bionpc.o | $(LIBDIR)
	ar rcs $@ $<

$(BENCH_BIN): $(EXDIR)/benchmark_brains.c $(SRCDIR)/bionpc.o | $(BINDIR)
	$(CC) $(CFLAGS) $< $(SRCDIR)/bionpc.o -o $@ $(LDFLAGS)
	@if [ -f $@.com ]; then cp -f $@.com $@; fi

$(SIM_BIN): $(EXDIR)/wolf_sheep_sim.c $(SRCDIR)/bionpc.o | $(BINDIR)
	$(CC) $(CFLAGS) $< $(SRCDIR)/bionpc.o -o $@ $(LDFLAGS)
	@if [ -f $@.com ]; then cp -f $@.com $@; fi

$(TEST_BIN): $(EXDIR)/test_io.c $(SRCDIR)/bionpc.o | $(BINDIR)
	$(CC) $(CFLAGS) $< $(SRCDIR)/bionpc.o -o $@ $(LDFLAGS)
	@if [ -f $@.com ]; then cp -f $@.com $@; fi

bench: $(BENCH_BIN)
	@mkdir -p ../.tmp
	TMPDIR=../.tmp HOME=../.tmp $(BENCH_BIN) 10000 8

sim: $(SIM_BIN)
	@mkdir -p ../.tmp
	TMPDIR=../.tmp HOME=../.tmp $(SIM_BIN) --wolves 1000 --sheep 4000 --steps 50 --threads 8

test: $(TEST_BIN)
	@mkdir -p ../.tmp
	TMPDIR=../.tmp HOME=../.tmp $(TEST_BIN)
	@echo "All Fil-C Cosmopolitan tests passed successfully!"

# Native GCC build target for performance cross-comparison
native:
	gcc -O3 -Wall -Wextra -Iinclude $(SRCDIR)/bionpc.c $(EXDIR)/benchmark_brains.c -o $(BINDIR)/bionpc_bench_gcc -lm -lpthread

clean:
	rm -rf $(SRCDIR)/*.o $(BINDIR) $(LIBDIR) ../.tmp
