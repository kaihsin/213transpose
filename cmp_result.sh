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
        
        #if [ $i -le 16 ]; then
        #    continue
        #fi
        
        echo "Case $i"
        echo "./test_213inplace ${PARAMS[$i]} > ./testcaes/large/res_case$i.out"
        ./test_213inplace ${PARAMS[$i]} ./testcase/large/res_case$i.out
		diff ./testcase/large/ans_case$i.out ./testcase/large/res_case$i.out
		rm -f ./testcase/large/res_case$i.out
done
