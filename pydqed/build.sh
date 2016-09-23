# Due to problems with linux32 gcc, you must first install gcc 4.8.2 onto your root anaconda build.  Then physically copy over the library files
# See http://github.com/conda/conda-recipes/issues/510

# Uncomment the following lines for linux-32
# rm ${PREFIX}/lib/gcc -r
# mkdir $PREFIX/lib/gcc
# cp -r (insert_base_anaconda_path_here)/lib/gcc/* ${PREFIX}/lib/gcc/

export CC=${PREFIX}/bin/gcc
export CXX=${PREFIX}/bin/g++
export F77=${PREFIX}/bin/gfortran
export F90=${PREFIX}/bin/gfortran
make
$PYTHON setup.py install

# uncomment the following lines to get the build to work in Mac OSX El Capitan, due to new issues relating to rpath

# install_name_tool -change @rpath/libgfortran.3.dylib ${PREFIX}/lib/libgfortran.3.dylib ${PREFIX}/lib/python2.7/site-packages/pydqed.so
# install_name_tool -change @rpath/libgfortran.3.dylib ${PREFIX}/lib/libgfortran.3.dylib ${SRC_DIR}/pydqed.so

# install_name_tool -change @rpath/./libquadmath.0.dylib ${PREFIX}/lib/libquadmath.0.dylib ${PREFIX}/lib/python2.7/site-packages/pydqed.so
# install_name_tool -change @rpath/./libquadmath.0.dylib ${PREFIX}/lib/libquadmath.0.dylib ${SRC_DIR}/pydqed.so

$PYTHON -c 'from pydqed import __version__; print __version__' > ${SRC_DIR}/__conda_version__.txt
