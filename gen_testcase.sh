#!/bin/bash

# $d1 $d2 $d3 $size_of_data_type
PARAMS=("1425 1000 1000 4"\
		"2 178125000 2 4"\
        "32 695800 32 4"\
		"33 695800 31 4"\
		"695800 31 33 4"\
        "178125000 2 2 4"\
		"1000 999 1425 4"\
        "32 32 695800 4"\
		"2 8 44531250 4"\
        "2 2 178125000 4"\
)

if [ ! -d "./testcase" ]; then
    mkdir -p ./testcase
fi

for ((i=0;i<${#PARAMS[@]};++i)) ; do
        echo "Case $i"
        echo "./gen_ans ${PARAMS[$i]} > ./testcase/ans_case$i.out"
        ./gen_ans ${PARAMS[$i]} ./testcase/ans_case$i.out
    done