INC_DIR = include
OBJ_DIR = build
OBJSLIB = $(OBJ_DIR)/test_213inplace.o\
		$(OBJ_DIR)/transpose.o\
		$(OBJ_DIR)/introspect.o\
		$(OBJ_DIR)/rotate.o\
		$(OBJ_DIR)/permute.o\
		$(OBJ_DIR)/shuffle.o\
		$(OBJ_DIR)/skinny.o\
		$(OBJ_DIR)/memory_ops.o\
		$(OBJ_DIR)/smem_ops.o\
		$(OBJ_DIR)/save_array.o\
		$(OBJ_DIR)/gcd.o\
		$(OBJ_DIR)/reduced_math.o\
		$(OBJ_DIR)/cudacheck.o\
		$(OBJ_DIR)/tensor_util.o\
        $(OBJ_DIR)/util.o\

OBJSGEN = $(OBJ_DIR)/gen_ans.o \
		$(OBJ_DIR)/tensor_util.o

CC = g++
NVCC = nvcc
CFLAGS = -I$(INC_DIR) -std=c++14 -O3
NVCCFLAGS = -arch=sm_61 -rdc=true

all: mkdir test_213inplace

mkdir:
	mkdir -p $(OBJ_DIR)

test_cutt: test_cutt.o $(OBJ_DIR)/tensor_util.o $(OBJ_DIR)/cudacheck.o ../cutt/lib/libcutt.a
	nvcc $(NVCCFLAGS) -std=c++14 -O3 -L../cutt/lib -lcutt -o test_cutt test_cutt.o $(OBJ_DIR)/tensor_util.o $(OBJ_DIR)/cudacheck.o
test_cutt.o: src/test_cutt.cu ../cutt/include/cutt.h ./include/tensor_util.h ./include/cudacheck.h
	nvcc $(NVCCFLAGS) -std=c++14 -O3 -I../cutt/include -I./include -c src/test_cutt.cu	

test_213inplace: $(OBJSLIB)
	$(NVCC) $(NVCCFLAGS) $(OBJSLIB) -o $@
	
gen_ans: $(OBJSGEN)
	$(CC) $(CFLAGS) $(OBJSGEN) -o $@

$(OBJ_DIR)/%.o : src/%.cu
	$(NVCC) -c $(CFLAGS) $(NVCCFLAGS) -o $(OBJ_DIR)/$*.o $<
	$(NVCC) -M $(CFLAGS) $(NVCCFLAGS) $< >> $(OBJ_DIR)/$*.d

$(OBJ_DIR)/%.o : src/%.cpp
	$(CC) -c $(CFLAGS) -o $(OBJ_DIR)/$*.o $<
	$(CC) -M $(CFLAGS) $< >> $(OBJ_DIR)/$*.d

-include $(OBJSLIB:.o=.d)
    
clean_dep:
	rm -f $(OBJ_DIR)/*.d
clean:
	rm -f $(OBJ_DIR)/*.o $(OBJ_DIR)/*.d test_213inplace 
