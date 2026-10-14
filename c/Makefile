# BioNPC C / Cosmopolitan APE & Engine SDK Makefile
#
# Automatically uses the Fil-C Cosmopolitan APE compiler when available
# to produce 100% memory-safe Actually Portable Executables (.com), and
# falls back cleanly to standard Clang/GCC on any developer or console host.

DEFAULT_FILC := /home/phil/.gemini/antigravity/scratch/fil-c/build/bin/clang
ifeq ($(origin CC), default)
    ifneq ($(wildcard $(DEFAULT_FILC)),)
        CC := $(DEFAULT_FILC)
    else
        CC := cc
    endif
endif

HOST_CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -Iinclude
LDFLAGS ?= -lm

SRCDIR := src
INCDIR := include
EXDIR := examples
BINDIR := bin
LIBDIR := lib

OBJS := $(SRCDIR)/bionpc.o

BENCH_BIN  := $(BINDIR)/bionpc_bench.com
SIM_BIN    := $(BINDIR)/wolf_sheep_sim.com
TEST_BIN   := $(BINDIR)/test_io.com
HYBRID_BIN := $(BINDIR)/hybrid_demo.com
STATIC_LIB := $(LIBDIR)/libbionpc.a
SHARED_LIB := $(LIBDIR)/libbionpc.so

.PHONY: all clean test bench sim hybrid native shared

all: $(BENCH_BIN) $(SIM_BIN) $(TEST_BIN) $(HYBRID_BIN) $(STATIC_LIB) $(SHARED_LIB)

$(BINDIR):
	mkdir -p $(BINDIR)

$(LIBDIR):
	mkdir -p $(LIBDIR)

$(SRCDIR)/bionpc.o: $(SRCDIR)/bionpc.c $(INCDIR)/bionpc.h
	$(CC) $(CFLAGS) -c $< -o $@

$(STATIC_LIB): $(SRCDIR)/bionpc.o | $(LIBDIR)
	ar rcs $@ $<

$(SHARED_LIB): $(SRCDIR)/bionpc.c $(INCDIR)/bionpc.h | $(LIBDIR)
	$(HOST_CC) -O3 -fPIC -shared -Wall -Wextra -I$(INCDIR) $(SRCDIR)/bionpc.c -o $@ -lm -lpthread

$(BENCH_BIN): $(EXDIR)/benchmark_brains.c $(SRCDIR)/bionpc.o | $(BINDIR)
	@rm -f $@ $@.com
	$(CC) $(CFLAGS) $< $(SRCDIR)/bionpc.o -o $@ $(LDFLAGS)
	@if [ -f $@.com ]; then mv -f $@.com $@; fi

$(SIM_BIN): $(EXDIR)/wolf_sheep_sim.c $(SRCDIR)/bionpc.o | $(BINDIR)
	@rm -f $@ $@.com
	$(CC) $(CFLAGS) $< $(SRCDIR)/bionpc.o -o $@ $(LDFLAGS)
	@if [ -f $@.com ]; then mv -f $@.com $@; fi

$(TEST_BIN): $(EXDIR)/test_io.c $(SRCDIR)/bionpc.o | $(BINDIR)
	@rm -f $@ $@.com
	$(CC) $(CFLAGS) $< $(SRCDIR)/bionpc.o -o $@ $(LDFLAGS)
	@if [ -f $@.com ]; then mv -f $@.com $@; fi

$(HYBRID_BIN): $(EXDIR)/hybrid_demo.c $(SRCDIR)/bionpc.o | $(BINDIR)
	@rm -f $@ $@.com
	$(CC) $(CFLAGS) $< $(SRCDIR)/bionpc.o -o $@ $(LDFLAGS)
	@if [ -f $@.com ]; then mv -f $@.com $@; fi

bench: $(BENCH_BIN)
	@mkdir -p ../.tmp
	TMPDIR=../.tmp HOME=../.tmp $(BENCH_BIN) 10000 8

sim: $(SIM_BIN)
	@mkdir -p ../.tmp
	TMPDIR=../.tmp HOME=../.tmp $(SIM_BIN) --wolves 1000 --sheep 4000 --steps 50 --threads 8

hybrid: $(HYBRID_BIN)
	@mkdir -p ../.tmp
	TMPDIR=../.tmp HOME=../.tmp $(HYBRID_BIN)

test: $(TEST_BIN)
	@mkdir -p ../.tmp
	TMPDIR=../.tmp HOME=../.tmp $(TEST_BIN)
	@echo "All BioNPC C / Fil-C Cosmopolitan tests passed successfully!"

# Native GCC build target for performance cross-comparison
native: | $(BINDIR)
	$(HOST_CC) -O3 -Wall -Wextra -Iinclude $(SRCDIR)/bionpc.c $(EXDIR)/benchmark_brains.c -o $(BINDIR)/bionpc_bench_gcc -lm -lpthread

clean:
	rm -rf $(SRCDIR)/*.o $(BINDIR) $(LIBDIR) ../.tmp
