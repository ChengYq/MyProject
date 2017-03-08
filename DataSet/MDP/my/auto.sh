f="my_kc1"
python ./convertToSVM.py "${f}_corr.arff" "${f}_corr"
python ./convertToSVM.py "${f}_featured.arff" "${f}_featured"
python ./convertToSVM.py "${f}_origin.arff" "${f}_origin"
python ./convertToSVM.py "${f}_test.arff" "${f}_test"


python /home/chyq/Download/libsvm-3.22/tools/easy.py "${f}_origin" "${f}_test"
python /home/chyq/Download/libsvm-3.22/tools/easy.py "${f}_featured" "${f}_test"
python /home/chyq/Download/libsvm-3.22/tools/easy.py "${f}_corr" "${f}_test"


# mv "$f.train" /home/chyq/Download/libsvm-3.22/tools
#mv "$f.test" /home/chyq/Download/libsvm-3.22/tools
#cd /home/chyq/Download/libsvm-3.22/tools
#python easy.py "$f.train" "$f.test"
#gedit ./"$f.test.predict"
