#!/bin/bash

# $d1 $d2 $d3 $size_of_data_type

PARAMS=("34 34 1189446 4"\
        "34 68 594723 4"\
        "34 136 297361 4"\
        "34 272 148680 4"\
        "34 544 74340 4"\
        "34 1088 37170 4"\
        "34 2176 18585 4"\
        "34 4352 9292 4"\
        "34 8704 4646 4"\
        "34 17408 2323 4"\
        "34 34816 1161 4"\
        "34 69632 580 4"\
        "34 139264 290 4"\
        "34 278528 145 4"\
        "34 557056 72 4"\
        "34 1114112 36 4"
)


if [ ! -d "./testcase" ]; then
    mkdir -p ./testcase
fi

for ((i=0;i<${#PARAMS[@]};++i)) ; do
        echo "Case $i"
        echo "./cutt ${PARAMS[$i]} > ./testcase/cutt_case$i.out"
        ./test_cutt ${PARAMS[$i]} ./testcase/cutt_case$i.out
    done