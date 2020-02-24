#!/bin/sh

test_path=$1
tag=$2
image=$3
echo $test_path
echo AHKHAKHAKHAKHAKHAKJHAKHKALHAKJHKAHKAHKLAJHKAHKAHKAHKAHLk
echo $pwd
docker run -v ${test_path}:/opt/ml --rm "${image}:${tag}" train
