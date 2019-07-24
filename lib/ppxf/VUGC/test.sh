#!/bin/zsh

for file in ./*.pdf; 
do 
	open $file 
	read value;
	printf "`echo $file`, `echo $value`\n" >> vals
	kill `pgrep Preview`
	
	if [ $value -eq 1 ]
	then
		folder="good"
	elif [ $value -eq 0 ]
	then
		folder="bad"
	elif [ $value -eq 2 ]
	then
		folder="inspect"
	else
		folder="uncertain"
	fi
	
	mv $file /Users/kmohamad/Documents/GitHub/SIP2019/lib/ppxf/VDGC/$folder/$file
done
