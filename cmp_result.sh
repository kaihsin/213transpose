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

PARAMS2=(        "343750000 2 2 4"\
        "171875000 2 4 4"\
        "85937500 2 8 4"\
        "42968750 2 16 4"\
        "21484375 2 32 4"\
        "10742187 2 64 4"\
        "5371093 2 128 4"\
        "2685546 2 256 4"\
        "1342773 2 512 4"\
        "671386 2 1024 4"\
        "335693 2 2048 4"\
        "167846 2 4096 4"\
        "83923 2 8192 4"\
        "41961 2 16384 4"\
        "20980 2 32768 4"\
        "10490 2 65536 4"\
        "5245 2 131072 4"\
        "2622 2 262144 4"\
        "1311 2 524288 4"\
        "655 2 1048576 4"\
        "327 2 2097152 4"\
        "163 2 4194304 4"\
        "81 2 8388608 4"\
        "40 2 16777216 4"\
        "20 2 33554432 4"\
        "10 2 67108864 4"\
        "5 2 134217728 4"\
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
        
        #if [ $i -le 16 ]; then
        #    continue
        #fi
        
        echo "Case $i"
        echo "./test_213inplace ${PARAMS[$i]}"
        nvprof --metrics achieved_occupancy ./test_213inplace ${PARAMS[$i]}
        #echo "./test_213inplace ${PARAMS[$i]} > ~/testcaes/large/res_case$i.out"
        #./test_213inplace ${PARAMS[$i]} ~/testcase/large/res_case$i.out
		#diff ~/testcase/large/ans_case$i.out ~/testcase/large/res_case$i.out
		#rm -f ~/testcase/large/res_case$i.out
done
