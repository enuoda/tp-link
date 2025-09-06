CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -flto -DNDEBUG
DEBUG_FLAGS = -g -O0 -Wall -Wextra -fsanitize=address

all: tepws_oo tepws_functional

tepws_oo.x: tepws_oo.cpp
	$(CXX) $(CXXFLAGS) -o $@ $

tepws_functional.x: tepws_functional.cpp
	$(CXX) $(CXXFLAGS) -o $@ $

debug: tepws_oo_debug tepws_functional_debug

tepws_oo_debug.x: tepws_oo.cpp
	$(CXX) $(DEBUG_FLAGS) -o $@ $

tepws_functional_debug.x: tepws_functional.cpp
	$(CXX) $(DEBUG_FLAGS) -o $@ $

clean:
	rm -f tepws_oo.x tepws_functional.x tepws_oo_debug.x tepws_functional_debug.x

.PHONY: all debug clean