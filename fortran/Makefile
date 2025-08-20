# =========================
# pyGREAT Shim Build (inside pygreat/fortran)
# =========================

# ---- Toolchain ----
FC          := gfortran
#make FC=mpif90
# ---- GREAT locations ----
# Adjust if your libgreat.a / module .mod files live elsewhere
GREAT_ROOT  ?= /Users/atorres/Codes/SN_mode_analysis/mode_analysis_bc/pygreat
GREAT_INC   := $(GREAT_ROOT)/include
GREAT_LIB   := $(GREAT_ROOT)/lib/libgreat.a

# ---- Local layout (this Makefile lives in pygreat/fortran) ----
SRCDIR      := .
OBJDIR      := ./obj
OUTLIBDIR   ?= ./lib

# ---- Platform detection ----
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SHLIB_EXT   := dylib
  SHLIB_FLAG  := -dynamiclib
  EXTRA_LDFLAGS := -Wl,-install_name,@rpath/libpygreat.$(SHLIB_EXT)
  # Default OpenBLAS location on Apple Silicon/Homebrew:
  OPENBLAS_HOME ?= /opt/homebrew/opt/openblas
  ifneq ("$(wildcard $(OPENBLAS_HOME)/lib)","")
    DEFAULT_LAPACKLIBS ?= -L$(OPENBLAS_HOME)/lib -lopenblas
  else
    DEFAULT_LAPACKLIBS ?= -lopenblas
  endif
else
  SHLIB_EXT   := so
  SHLIB_FLAG  := -shared
  EXTRA_LDFLAGS :=
  DEFAULT_LAPACKLIBS ?= -lopenblas
endif

# ---- Flags ----
FFLAGS      ?= -O3 -g -fPIC -ffree-line-length-none -J$(GREAT_INC) -I$(GREAT_INC)

# Preserve env LDFLAGS but PREPEND our shared-lib flags so we always build a library
ENV_LDFLAGS := $(LDFLAGS)
override LDFLAGS := $(SHLIB_FLAG) $(EXTRA_LDFLAGS) $(ENV_LDFLAGS)

# External libs (defaults to OpenBLAS; override if needed)
LAPACKLIBS  ?= $(DEFAULT_LAPACKLIBS)
MPILIBS     ?=
EXTLIBS     ?=
LDLIBS      := $(LAPACKLIBS) $(MPILIBS) $(EXTLIBS)

# ---- Targets ----
TARGET      := $(OUTLIBDIR)/libpygreat.$(SHLIB_EXT)
OBJS        := $(OBJDIR)/pygreat_shim.o

.PHONY: all clean dirs print

all: $(TARGET)

dirs:
	@mkdir -p $(OBJDIR) $(OUTLIBDIR)

$(TARGET): dirs $(OBJS) $(GREAT_LIB)
	$(FC) $(LDFLAGS) -o $@ $(OBJS) $(GREAT_LIB) $(LDLIBS)
	@echo "Built: $@"
	@ls -lh $@ || true

$(OBJDIR)/pygreat_shim.o: $(SRCDIR)/pygreat_shim.f90 | dirs
	$(FC) $(FFLAGS) -c $< -o $@

clean:
	@rm -f $(OBJDIR)/*.o $(OBJDIR)/*.mod $(TARGET)

print:
	@echo "FC            = $(FC)"
	@echo "UNAME_S       = $(UNAME_S)"
	@echo "GREAT_ROOT    = $(GREAT_ROOT)"
	@echo "GREAT_INC     = $(GREAT_INC)"
	@echo "GREAT_LIB     = $(GREAT_LIB)"
	@echo "OUTLIBDIR     = $(OUTLIBDIR)"
	@echo "SHLIB_EXT     = $(SHLIB_EXT)"
	@echo "FFLAGS        = $(FFLAGS)"
	@echo "LDFLAGS       = $(LDFLAGS)"
	@echo "LAPACKLIBS    = $(LAPACKLIBS)"
	@echo "MPILIBS       = $(MPILIBS)"
	@echo "EXTLIBS       = $(EXTLIBS)"
