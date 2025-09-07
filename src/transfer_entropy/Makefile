CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -flto -DNDEBUG
DEBUG_FLAGS = -g -O0 -Wall -Wextra -fsanitize=address

all: tepws_functional.x

tepws_functional.x:
	$(CXX) $(CXXFLAGS) -o $@ tepws_functional.cpp gillespie.cpp gillespie.hpp

debug: tepws_functional_debug.x

tepws_functional_debug.x:
	$(CXX) $(DEBUG_FLAGS) -o $@ tepws_functional.cpp gillespie.cpp gillespie.hpp

clean:
	rm -f tepws_functional.x tepws_functional_debug.x

.PHONY: all debug clean