count=60;
while [ $count -le 119 ];
do
  $(printf "/home/graviti/WorkingMain/my_scalp/test %06d_10.png result_$count" $count);
  count=`expr $count + 1`;
done

