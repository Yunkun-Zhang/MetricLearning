
for k in 5, 7, 9, 11, 13
do
  # lmnn, try using original 2048 dim data and reduced 49-dim data
  python main.py -m lmnn -k $k -d 49
  python main.py -m lmnn -k $k -d 49 -r

  # nca, just like lmnn, but without k
  python main.py -m nca -k $k -d 49
  python main.py -m nca -k $k -d 49 -r

  # lfda
  for n in 16, 32, 64, 128, 256, 512
  do
    python main.py -m lfda -k $k -d $n
  done

  # itml
  python main.py -m itml -k $k -r

  # sdml
  python main.py -m itml -k $k -r

  # rca
  python main.py -m rca -k $k -r

  # lsml
  python main.py -m lsml -k $k

  # mmc
  python main.py -m mmc -k $k -r

  # mlkr, maybe need to use less then 1000 samples? maybe commment this when first try to run.
  python main.py -m mlkr -k $k -r
done




