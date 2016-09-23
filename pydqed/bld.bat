cp "%PREFIX%\..\..\libs\libpython27.a" "%PREFIX%\libs"
cp "%PREFIX%\..\..\Lib\distutils\distutils.cfg" "%PREFIX%\Lib\distutils\distutils.cfg

gfortran -fPIC -O3 -c dqed.f90 -o dqed.o
ar rcs libdqed.a dqed.o

"%PYTHON%" setup.py build_ext --build-lib . --build-temp build --pyrex-c-in-temp --compiler=mingw32

"%PYTHON%" setup.py install


"%PYTHON%" -c "from pydqed import __version__; print __version__" > "%SRC_DIR%/__conda_version__.txt"