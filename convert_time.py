#!/usr/bin/python3
import sys

with open(sys.argv[1]) as f:
    data = f.readlines()
    for line in data:
        minutes = float(line.split('m')[0])
        seconds = float((line.split('m'))[1].strip("s\n"))
        #print(minutes, seconds)
        #out_data.append(str((minutes * 60) + seconds) + '\n')
        print((minutes * 60) + seconds)
