# Shortcuts over CMake + CTest. Windows (Strawberry): `gmake <target>`; Linux: `make <target>`.
#   gmake                          build and run every test
#   gmake test T=^Image            build and run the tests whose name (Suite.Test) matches regex T
#   gmake build                    build only
#   gmake release                  optimised build (no asserts) in build-release/, for timing real runs
#   gmake clean                    delete the build folders (the next build reconfigures, slow)

BUILD_DIR := build
RELEASE_DIR := build-release
T ?= .

.PHONY: all build test configure release clean

all: test

$(BUILD_DIR)/CMakeCache.txt:
	cmake -S . -B $(BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=Debug

configure: $(BUILD_DIR)/CMakeCache.txt

build: configure
	cmake --build $(BUILD_DIR)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure -R "$(T)"

$(RELEASE_DIR)/CMakeCache.txt:
	cmake -S . -B $(RELEASE_DIR) -G Ninja -DCMAKE_BUILD_TYPE=Release

release: $(RELEASE_DIR)/CMakeCache.txt
	cmake --build $(RELEASE_DIR)

clean:
	cmake -E rm -rf $(BUILD_DIR) $(RELEASE_DIR)
