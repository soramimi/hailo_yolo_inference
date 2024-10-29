all:
	g++ async_infer_basic_example.cpp -lhailort

clean:
	-rm a.out
	-rm *.log
	-rm *.o *.d
