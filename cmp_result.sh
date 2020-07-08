#!/bin/bash

# $d1 $d2 $d3 $size_of_data_type

PARAMS=("32 32 695800 4 "\
		"32 1024 21742 4"\
		"32 2048 10871 4"
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
        echo "Case $i"
        echo "./test_213inplace ${PARAMS[$i]} > ./testcase/res_case$i.out"
        ./test_213inplace ${PARAMS[$i]} ./testcase/res_case$i.out
		diff ./testcase/ans_case$i.out ./testcase/res_case$i.out
		rm -f ./testcase/res_case$i.out
done
