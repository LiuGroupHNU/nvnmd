

echo "INFO: start compile"
date

# GET TENSORFLOW LIB FLAG
#TF_CFLAGS='-I/home/mph/InstallPackage/Application/Anaconda/lib/python3.7/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=1'
#TF_LFLAGS='-L/home/mph/InstallPackage/Application/Anaconda/lib/python3.7/site-packages/tensorflow -l:libtensorflow_framework.so.1'
TF_CFLAGS=`python -c 'import tensorflow as tf; print( (" ".join(tf.sysconfig.get_compile_flags()) ))'`
TF_LFLAGS=`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'`
echo 
echo "INFO: get the FLAGS from python"
echo $TF_CFLAGS
echo $TF_LFLAGS 
echo

# COMPILE

mkdir -p build
cd build

echo 
echo "INFO: compile libdpOp.so"
echo

name="libdpOp"
rm -r ./*.*
cp ../src/dp/*.* ./
g++ -std=c++11 -shared *.cpp *.cc *.h -o ${name}.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -L./ -O2 
cp *.so ../

echo 
echo "INFO: compile libdpGradOp.so"
echo

name="libdpGradOp"
rm -r ./*.*
cp ../src/dp_grad/*.* ./
g++ -std=c++11 -shared *.cc -o ${name}.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -L./ -O2 
cp *.so ../


echo 
echo "INFO: compile libmzOp.so"
echo

name="libmzOp"
rm -r ./*.*
cp ../src/mz/*.* ./
g++ -std=c++11 -shared *.cc -o ${name}.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -L./ -O2 
cp *.so ../

cd ../

echo "INFO: finish compile"
date
