f="my_kc1"
python ./convertToSVM.py "${f}_corr.arff" "${f}_corr"
python ./convertToSVM.py "${f}_featured.arff" "${f}_featured"
python ./convertToSVM.py "${f}_origin.arff" "${f}_origin"
python ./convertToSVM.py "${f}_test.arff" "${f}_test"
python ./convertToSVM.py "${f}_info.arff" "${f}_info"




echo "origin..."
python /home/chyq/Download/libsvm-3.22/tools/easy.py "${f}_origin" "${f}_test"
echo "featured..."
python /home/chyq/Download/libsvm-3.22/tools/easy.py "${f}_featured" "${f}_test"
echo "corr..."
python /home/chyq/Download/libsvm-3.22/tools/easy.py "${f}_corr" "${f}_test"
echo "info..."
python /home/chyq/Download/libsvm-3.22/tools/easy.py "${f}_info" "${f}_test"

echo "origin_weighted..."
python /home/chyq/Download/libsvm-3.22/tools/easy2.py "${f}_origin" "${f}_test"
echo "featured_weighted..."
python /home/chyq/Download/libsvm-3.22/tools/easy2.py "${f}_featured" "${f}_test"
echo "corr_weighted..."
python /home/chyq/Download/libsvm-3.22/tools/easy2.py "${f}_corr" "${f}_test"
echo "info_weighted..."
python /home/chyq/Download/libsvm-3.22/tools/easy2.py "${f}_info" "${f}_test"





# mv "$f.train" /home/chyq/Download/libsvm-3.22/tools
#mv "$f.test" /home/chyq/Download/libsvm-3.22/tools
#cd /home/chyq/Download/libsvm-3.22/tools
#python easy.py "$f.train" "$f.test"
#gedit ./"$f.test.predict"
