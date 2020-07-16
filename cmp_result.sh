#!/bin/bash

# $d1 $d2 $d3 $size_of_data_type


PARAMS=(    "2 343750000 2 4"\
	"2 171875000 4 4"\
	"2 85937500 8 4"\
	"2 42968750 16 4"\
	"2 21484375 32 4"\
	"2 10742187 64 4"\
	"2 5371093 128 4"\
	"2 2685546 256 4"\
	"2 1342773 512 4"\
	"2 671386 1024 4"\
	"2 335693 2048 4"\
	"2 167846 4096 4"\
	"2 83923 8192 4"\
	"2 41961 16384 4"\
	"2 20980 32768 4"\
	"2 10490 65536 4"\
	"2 5245 131072 4"\
	"2 2622 262144 4"\
	"2 1311 524288 4"\
	"2 655 1048576 4"\
	"2 327 2097152 4"\
	"2 163 4194304 4"\
	"2 81 8388608 4"\
	"2 40 16777216 4"\
	"2 20 33554432 4"\
	"2 10 67108864 4"\
	"2 5 134217728 4"\
	"2 2 268435456 4"\
)



#PARAMS=("2 178125000 2 4"\
#        "31 695800 31 4"\
#		"30 695800 29 4"\
#		"695800 28 30 4"\
#        "178125000 2 2 4"\
#)

if [ ! -d "./testcase" ]; then
    mkdir -p ./testcase
fi

for ((i=0;i<${#PARAMS[@]};++i)) ; do
        #if [ $i -gt 15 ]; then
        #    continue
        #fi
        
        if [ $i -le 16 ]; then
            continue
        fi
        
        echo "Case $i"
        echo "./test_213inplace ${PARAMS[$i]} > ./testcaes/fix_d1/res_case$i.out"
        ./test_213inplace ${PARAMS[$i]} ./testcase/fix_d1/res_case$i.out
		diff ./testcase/fix_d1/ans_case$i.out ./testcase/fix_d1/res_case$i.out
		rm -f ./testcase/fix_d1/res_case$i.out
done
