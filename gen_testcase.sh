#!/bin/bash

# $d1 $d2 $d3 $size_of_data_type
PARAMS=("2 178125000 2 4"\
        "31 695800 31 4"\
		"30 695800 29 4"\
		"695800 28 30 4"\
        "178125000 2 2 4"\
        "31 31 695800 4"\
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