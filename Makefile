INC_DIR = include
SRC_DIR = src
OBJ_DIR = build
LIB_DIR = lib
OBJSLIB = $(OBJ_DIR)/transpose.o\
		$(OBJ_DIR)/introspect.o\
		$(OBJ_DIR)/rotate.o\
		$(OBJ_DIR)/permute.o\
		$(OBJ_DIR)/shuffle.o\
		$(OBJ_DIR)/col_shuffle.o\
		$(OBJ_DIR)/skinny.o\
		$(OBJ_DIR)/memory_ops.o\
		$(OBJ_DIR)/smem_ops.o\
		$(OBJ_DIR)/save_array.o\
		$(OBJ_DIR)/gcd.o\
		$(OBJ_DIR)/reduced_math.o\
		$(OBJ_DIR)/cudacheck.o\
		$(OBJ_DIR)/tensor_util.o\
        $(OBJ_DIR)/util.o
OBJTEST = $(OBJ_DIR)/test_213inplace.o
		
CC = g++
NVCC = nvcc
CPPFLAGS = -I$(INC_DIR) -std=c++14 -O3
NVCCFLAGS = -arch=sm_61 -rdc=true

all: mkdir lib/libinplacett.a test_213inplace

mkdir:
	mkdir -p $(OBJ_DIR)

test_213inplace: lib/libinplacett.a $(OBJTEST)
	$(NVCC) $(NVCCFLAGS) -L$(LIB_DIR) -linplacett -o $@ $^

lib/libinplacett.a: $(OBJSLIB)
	mkdir -p $(LIB_DIR)
	rm -f lib/libinplacett.a
	ar -cvq lib/libinplacett.a $(OBJSLIB)
	
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(OBJ_DIR)/%.d
	$(NVCC) -c $(CPPFLAGS) $(NVCCFLAGS) -o $@ $<

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(OBJ_DIR)/%.d
	$(CC) -c $(CPPFLAGS) -o $@ $<
	
$(OBJ_DIR)/%.d: $(SRC_DIR)/%.cu
	@echo "$(NVCC) -MM $(CPPFLAGS) $(NVCCFLAGS) $< > $@"
	@$(NVCC) -MM $(CPPFLAGS) $(NVCCFLAGS) $< > $@.tmp
	@sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.tmp > $@
	@rm -f $@.tmp

$(OBJ_DIR)/%.d: $(SRC_DIR)/%.cpp
	@echo "$(CC) -MM $(CPPFLAGS) $< > $@"
	@$(CC) -MM $(CPPFLAGS) $< > $@.tmp
	@sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.tmp > $@
	@rm -f $@.tmp
	
-include $(OBJSLIB:.o=.d)
	
clean:
	rm -f $(OBJ_DIR)/* lib/* $(EXEC)
