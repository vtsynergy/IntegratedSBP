PATCHES=-I./patches
TCLAP=-I./tclap-1.2.2/include
INCLUDES=-I./include -I./include/blockmodel -I./include/blockmodel/sparse -I./hungarian $(PATCHES) $(TCLAP)
CC=g++ -std=c++17 -O3 -no-pie
MPICC=mpic++ -std=c++17 -O3 -no-pie
DEBUGCC=mpic++ -std=c++17 -no-pie -g3 -fsanitize=address
FILES=main.cpp graph.cpp utils.cpp sbp.cpp block_merge.cpp common.cpp finetune.cpp partition.cpp
FILES+=blockmodel/blockmodel.cpp blockmodel/dist_blockmodel.cpp blockmodel/blockmodel_triplet.cpp
FILES+=blockmodel/sparse/boost_mapped_matrix.cpp blockmodel/sparse/dict_matrix.cpp
FILES+=blockmodel/sparse/dist_dict_matrix.cpp blockmodel/sparse/dict_transpose_matrix.cpp evaluate.cpp
FILES+=hung.o

hung.o:
	$(CC) -o hung.o -c hungarian/hungarian.cpp $(PATCHES)

main: hung.o main.cpp
	$(MPICC) -fopenmp -o main $(FILES) $(INCLUDES) -lstdc++fs

debugmain: hung.o main.cpp
	$(DEBUGCC) -fopenmp -o main $(FILES) $(INCLUDES) -lstdc++fs

runmain: main
	./main -d ../graph_challenge/data
	rm main hung.o
