all:
	./glslang compute.comp
	g++ -DLINUX NUNEC.c -lGL -lX11  transform.cpp libGLEW.a
	optirun ./a.out
