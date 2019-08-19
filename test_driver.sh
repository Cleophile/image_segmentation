count=60;
while [ $count -le 119 ];
do
  $(printf "/home/graviti/WorkingMain/my_scalp/test ./%06d_10.png /home/graviti/下载/scalp_result/result_%06d.jpg /home/graviti/下载/scalp_text/result_%06d.txt" $count $count $count);
  count=`expr $count + 1`;
done

