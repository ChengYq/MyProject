f="my_kc1"
python ./convertToSVM.py "$f.arff" "$f.train" "$f.test"
mv "$f.train" /home/chyq/Download/libsvm-3.22/tools
mv "$f.test" /home/chyq/Download/libsvm-3.22/tools
cd /home/chyq/Download/libsvm-3.22/tools
python easy.py "$f.train" "$f.test"
gedit ./"$f.test.predict"
