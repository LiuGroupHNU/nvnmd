

tensorflow_root='/home/mph/InstallPackage/Package/deepmd/tensorflow/tensorflow/bin'
deepmd_root='/home/mph/InstallPackage/Package/deepmd/deepmd-kit/bin'

rm -r build
mkdir build
cd build

cmake-3.10.0 -DTENSORFLOW_ROOT=$tensorflow_root -DCMAKE_INSTALL_PREFIX=$deepmd_root ..

make -j 6
make install

cd ../

echo 
ls $deepmd_root/bin

echo
ls $deepmd_root/lib