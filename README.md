### Neural Network Project
This is a basic hobby neural network library, written entirely from scratch except for the help of a matrix/vector library (eigen)  
It is currently set up with an example project - recognizing hand-drawn digits from the MNIST dataset (found in the 'training' directory)

compute.cpp is not currently used, but is planned so that training can be performed on the GPU

### Build Instructions
Running ``make`` should build and run the example project, but requires SFML for the interactive window demo


### TODO
Implement GPU feedforward and backpropogation  
flesh out Network class with new activ/cost functions and learn theory behind them  
saved networks should store activ/cost functions and have nice header  
improve readability/comments/structure of dnetwork() and train()  
ensure all backpropogation calculations are working correctly/accurately  
mini-batches should be randomly chosen?  
implement softmax and log-liklihood properly?  

other optimization algorithms? Differing ways of doing gradient descent such as using 2nd deriv or newton's method? Converge on local minima faster?  
Regarding generating new examples using gradient descent on input:  
regularization?  
stopping criterion?  

Other types of networks/layers/architectures:  
recurrent  
convolutional  
GANs (generator and discriminator)  

Project Ideas:  
AI evolution project  
chess ai that trains against itself  