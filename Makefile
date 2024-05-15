default: run

run: neural
	./neural

neural: main.cpp network.cpp dataset.cpp
	g++ -O3 -I eigen $^ -o $@ -Wall -pthread -lsfml-graphics -lsfml-window -lsfml-system
	
%.cpp: %.hpp

clean:
	rm neural