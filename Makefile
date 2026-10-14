CC = cc
PYTHON = python3
CPPFLAGS = -Iinclude
CFLAGS = -std=c11 -O3 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off
BUILD = build
BLAS ?= 0
ENGINE = src/tiny_infer.c src/blas_backend.c
HEADERS = include/tiny_infer.h src/blas_backend.h
OBJECTS = $(BUILD)/tiny_infer.o $(BUILD)/blas_backend.o
ifeq ($(BLAS),1)
  ifeq ($(shell uname -s),Darwin)
    CPPFLAGS += -DTI_ENABLE_BLAS -DACCELERATE_NEW_LAPACK
    LDLIBS = -framework Accelerate
  else
    CPPFLAGS += -DTI_ENABLE_BLAS -DTI_BLAS_OPENBLAS
    LDLIBS = -lopenblas
  endif
endif

.PHONY: all demo test sanitize mutation build-check publication-check bench clean FORCE
all: $(BUILD)/test_batch $(BUILD)/test_engine $(BUILD)/libtiny_infer.so $(BUILD)/benchmark $(BUILD)/demo
$(BUILD):
	mkdir -p $@
FORCE:
# Always rebuild these small objects, including changes to BLAS/compiler flags.
$(BUILD)/%.o: src/%.c $(HEADERS) Makefile FORCE | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -c $< -o $@
$(BUILD)/libtiny_infer.so: $(OBJECTS)
	$(CC) -shared $(OBJECTS) $(LDLIBS) -lm -o $@
$(BUILD)/test_engine: tests/test_engine.c $(OBJECTS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ $(LDLIBS) -lm -o $@
$(BUILD)/test_batch: tests/test_batch.c $(OBJECTS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ $(LDLIBS) -lm -o $@
$(BUILD)/benchmark: tools/benchmark.c $(OBJECTS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ $(LDLIBS) -lm -o $@
$(BUILD)/demo: examples/demo.c $(OBJECTS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ $(LDLIBS) -lm -o $@
demo: $(BUILD)/demo
	./$(BUILD)/demo
test: $(BUILD)/test_batch $(BUILD)/test_engine $(BUILD)/libtiny_infer.so
	./$(BUILD)/test_engine
	./$(BUILD)/test_batch
	$(PYTHON) tests/test_batch_reference.py $(BUILD)/libtiny_infer.so
	$(PYTHON) -O tests/test_batch_reference.py $(BUILD)/libtiny_infer.so
	$(PYTHON) tests/test_reference.py $(BUILD)/libtiny_infer.so
	$(PYTHON) -O tests/test_reference.py $(BUILD)/libtiny_infer.so
	$(PYTHON) tests/check_no_heap.py $(BUILD)/tiny_infer.o
	$(PYTHON) tests/test_reporting.py
	$(PYTHON) -O tests/test_reporting.py
	$(PYTHON) tests/test_publication.py
	$(PYTHON) tests/test_documentation.py
	$(PYTHON) -O tests/test_documentation.py
	$(PYTHON) -O tests/test_publication.py
sanitize: | $(BUILD)
	$(CC) $(CPPFLAGS) -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -fsanitize=address,undefined -fno-omit-frame-pointer tests/test_engine.c $(ENGINE) $(LDLIBS) -lm -o $(BUILD)/test_sanitize
	UBSAN_OPTIONS=halt_on_error=1 ./$(BUILD)/test_sanitize
	$(CC) $(CPPFLAGS) -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -fsanitize=address,undefined -fno-omit-frame-pointer tests/test_batch.c $(ENGINE) $(LDLIBS) -lm -o $(BUILD)/test_batch_sanitize
	UBSAN_OPTIONS=halt_on_error=1 ./$(BUILD)/test_batch_sanitize
mutation: | $(BUILD)
	$(PYTHON) tests/test_mutations.py --cc "$(CC)" --blas $(BLAS)
build-check:
	$(PYTHON) tests/check_build_flags.py --cc "$(CC)"
publication-check:
	$(PYTHON) tools/check_publication.py
	$(PYTHON) tools/check_documentation.py
	$(PYTHON) tools/verify_report.py results/published/apple-m5-pro-20261014
bench:
	$(PYTHON) tools/run_benchmark.py --cc "$(CC)" --cflags "$(CFLAGS)" --blas $(BLAS)
clean:
	rm -rf $(BUILD)
