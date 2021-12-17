
ct=0
for p in `ls`
do
l=`wc -l $p/coord.raw`
ct=`echo "$ct $l" | awk '{printf("%d",$1+$2)}'`
echo $ct $l
done

