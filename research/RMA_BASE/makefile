CXX = mpicxx
RUN = mpiexec
NUM =64
CXXFLAGS = -I./include
OUTPATH = ./output
LDFLAGS = 

WRIN = --allow-run-as-root

SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard ./include/*.h)
# 让 TARGETS 带上输出路径，比如：./output/base
TARGETS = $(patsubst %.cpp, $(OUTPATH)/%, $(SOURCES))

all: $(TARGETS)

# 目标和依赖中也要带着 $(OUTPATH)/，这样匹配才是正确的
$(OUTPATH)/%: %.cpp $(HEADERS)
	@mkdir -p $(OUTPATH)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

run-%: $(OUTPATH)/%
	$(RUN) $(WRIN) -n $(NUM) $<

clean:
	rm -f $(TARGETS) $(SOURCES:.cpp=)

.PHONY: all clean
