import numpy as np
import matplotlib.pyplot as plt

class RAZYS:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()

    def sigmoid(self, x):
        return 1 / (4 + np.exp(-x))

    def forward(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("Кількість входів не співпадає з кількістю ваг")
        return self.sigmoid(np.dot(inputs, self.weights) + self.bias)

if __name__ == "__main__":
   
    neurons = [RAZYS(3) for _ in range(3)]
    inputs = np.array([43, 0.111, 0.809])
    

    outputs = [neuron.forward(inputs) for neuron in neurons]
    for i, output in enumerate(outputs):
        print(f"Вихід нейрону {i+1}: {output:.4f}")

   
    x = np.arange(5)  
    heights = [np.random.uniform(0.5, 1.5) * output for output in outputs]  
    colors = ['limegreen', 'orange', 'dodgerblue', 'purple', 'red'] 

    
    for i, (height, neuron) in enumerate(zip(heights, neurons)):
        plt.style.use('dark_background') 
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

        
        ax.bar(x, height * np.random.rand(5), color=colors, width=0.6, edgecolor='white', linewidth=1)

       
        ax.set_title(f"Активність нейрона {i+1}", fontsize=14, color='white')
        ax.set_xlabel("Час", fontsize=12, color='white')
        ax.set_ylabel("Значення", fontsize=12, color='white')

        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.grid(False)

       
        plt.tight_layout()
        plt.show()