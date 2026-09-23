# MNIST Digit Classification with CNN

## Objective
Develop and train a convolutional neural network (CNN) in PyTorch to classify images from the MNIST dataset. The goal is to achieve high accuracy in distinguishing between the 10 digit classes.

## Model Architecture
### Input Specifications
- **Image size:** 28x28 pixels (grayscale images).

### Network Architecture
* **First Convolutional Layer:**
   - Input channels: 1 (grayscale image)
   - Output channels: 32
   - Kernel size: 5x5
   - Stride: 1
   - Padding: 2
* Activation Function: ReLU
* Pooling Layer:
     - Type: Max Pooling
     - Kernel size: 2x2
     - Stride: 2

* **Second Convolutional Layer:**
   - Input channels: 32
   - Output channels: 64
   - Kernel size: 5x5
   - Stride: 1
   - Padding: 2
* Activation Function: ReLU
* Pooling Layer:
     - Type: Max Pooling
     - Kernel size: 2x2
     - Stride: 2

* **Flatten Layer:**
   - Flatten the output from the previous pooling layer to feed into the fully connected layer.

* **First Fully Connected Layer:**
   - Input features: 7x7x64 = 3136
   - Output features: 1024
   - Activation Function: ReLU
* Dropout Layer:
     - Dropout probability: 0.5

* **Output Layer:**
   - Input features: 1024
   - Output features: 10 (corresponding to the 10 digits)



## Tasks:

* Implement the network using PyTorch’s frameworks. (You only need to modify the class Net(nn.Module) section within mnist_classify.py in week 4 and do not change the loss function output = F.log_softmax(x, dim=1))

  An example of convolutional neural networks with padding
```
self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
```
* Train the network for at least 5 epochs.
* Profile your code by changing the number of workers (from 1 worker to 8 workers) in the kwargs function:
  
Line 166:  cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}

Line 166:  cuda_kwargs = {'num_workers': 8, 'pin_memory': True, 'shuffle': True}

Remember to change your slurm script so that cpus-per-task = 8 instead of cpus-per-task = 1

## Deliverables:
* A brief description of the model architecture. (pdf format)
* Python code for the entire training pipeline: mnist_classify.py and your slurm script used for job submission. (Either with number_workers=1 or 8, but slurm job script needs to be consistent with your mnist_calssify.py) 
* Results of line profiler with different number of workers and your analysis of the results. (See github repo step 3 for sampled results from line_profiler).  (pdf format) 

Here is the reference code from week4 class excercise: 
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer:
        # - Input channels: 1 (e.g., grayscale images)
        # - Output channels: 32
        # - Kernel size: 3x3
        # - Stride: 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        
        # Second convolutional layer:
        # - Input channels: 32 (from conv1)
        # - Output channels: 64
        # - Kernel size: 3x3
        # - Stride: 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        
        # First dropout layer:
        # - Dropout probability: 25%
        # - Helps prevent overfitting by randomly zeroing some of the elements of the input tensor
        self.dropout1 = nn.Dropout(p=0.25)
        
        # Second dropout layer:
        # - Dropout probability: 50%
        # - Further regularizes the model during training
        self.dropout2 = nn.Dropout(p=0.5)
        
        # First fully connected (linear) layer:
        # - Input features: 9216
        #   (Assuming input image size is 28x28, after two conv layers and pooling)
        # - Output features: 128
        self.fc1 = nn.Linear(in_features=9216, out_features=128)
        
        # Second fully connected (linear) layer:
        # - Input features: 128
        # - Output features: 10 (e.g., number of classes for classification)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Pass input through the first convolutional layer
        x = self.conv1(x)
        # Apply ReLU activation function
        x = F.relu(x)
        
        # Pass the result through the second convolutional layer
        x = self.conv2(x)
        # Apply ReLU activation function
        x = F.relu(x)
        
        # Apply 2D max pooling with a kernel size of 2
        # This reduces the spatial dimensions by a factor of 2
        x = F.max_pool2d(x, kernel_size=2)
        
        # Apply the first dropout layer
        x = self.dropout1(x)
        
        # Flatten the tensor starting from the first dimension (excluding batch size)
        # This prepares the data for the fully connected layers
        x = torch.flatten(x, 1)
        
        # Pass through the first fully connected layer
        x = self.fc1(x)
        # Apply ReLU activation function
        x = F.relu(x)
        
        # Apply the second dropout layer
        x = self.dropout2(x)
        
        # Pass through the second fully connected layer
        x = self.fc2(x)
        
        # Apply log softmax activation function to obtain log probabilities for each class
        output = F.log_softmax(x, dim=1)
        
        return output
```






