#include "network.hpp"
#include "dataset.hpp"
//#include "compute.hpp"

#include <iostream>
#include <SFML/Graphics.hpp>
#define WIDTH 28
#define SCALE 20



using namespace Eigen;

int main() {
	//create network
	//784 neurons in input layer for each pixel of image, 2 layers of 100 neurons, then 10 neurons in final layer for each possible digit
	Network network({784, 100, 100, 10});
	network.randomize();
	//Network* network = Network::load("./numnet");

	//load datasets
	Dataset trainset;
	Dataset testset;

	//MNIST digits
	trainset.add_examples_bin("./training/train-images-idx3-ubyte", 16, 784, 784);
	trainset.add_labels_bin("./training/train-labels-idx1-ubyte", 8, 1, 10);
	testset.add_examples_bin("./training/t10k-images-idx3-ubyte", 16, 784, 784);
	testset.add_labels_bin("./training/t10k-labels-idx1-ubyte", 8, 1, 10);
	
	//train and score
	network.train(trainset, 10, 100, 0.1); //10 epochs with batches of size 100, and learning rate of 0.1
	std::cout << "Cost: " << network.cost(testset) << std::endl;
	std::cout << "Accuracy: " << network.accuracy(testset) << "%" << std::endl;
	
	//network->save("./numnet");
	


	//The code below creates a window that lets the user draw digits, which are fed through the trained network, displaying its output layer activations
	//Left click to draw. Right click to erase.
	sf::RenderWindow window(sf::VideoMode(WIDTH * SCALE + 100, WIDTH * SCALE), "Digit Recognizer", sf::Style::Titlebar | sf::Style::Close);
    sf::Texture texture;
    texture.create(28, 28);
    sf::Sprite sprite;
    sprite.setTexture(texture);
    sprite.setScale(sf::Vector2f(SCALE, SCALE));

	sf::Font font;
	font.loadFromFile("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf");
	sf::Text text;
	text.setFont(font);
	text.setCharacterSize(45);

    sf::Uint32* pixels = new sf::Uint32[WIDTH * WIDTH];
	VectorXd draw = testset.examples.at(1);
    for (int i = 0; i < WIDTH * WIDTH; i++) {
		uint8_t val = draw[i] * 255;
		pixels[i] = val | val << 8 | val << 16 | val << 24;
	}

	Eigen::VectorXd digit(784);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event))
            if (event.type == sf::Event::Closed) window.close();
		
		//draw to screen
        sf::Vector2i pos = sf::Mouse::getPosition(window);
		if (pos.x > WIDTH * SCALE || pos.y > WIDTH * SCALE) {}
        else if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) pixels[((pos.y/SCALE) * WIDTH + pos.x/SCALE)] = 0xffffffff;
        else if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) pixels[((pos.y/SCALE) * WIDTH + pos.x/SCALE)] = 0x0;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space)) {
			for (int i = 0; i < WIDTH * WIDTH; i++) pixels[i] = 0x0;
		}

		//convert pixel array to vector input
		for (int i = 0; i < 784; i++) {
			digit[i] = (double)(pixels[i] & 0xff) / 255;
		}

		//calculate digit
		VectorXd result = network.feedForward(digit);

		window.clear();
		texture.update((sf::Uint8*)pixels);
        window.draw(sprite);

		//show digit
		for (int i = 0; i < 10; i++) {
			char number[2] = {(char)(i + 48), 0};
			text.setString(number);
			text.setPosition(600, 5 + 55 * i);
			sf::Uint8 color = (int)(result[i] * (double)255);
			text.setFillColor(sf::Color(color, color, color, 0xff));
			window.draw(text);
		}

        window.display();
    }
}