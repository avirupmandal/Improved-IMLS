import numpy as np
import matplotlib.pyplot as plt
import os
def plot_weight_functions(m=10, kappa_epsilon=0.08, h=0.2, a=10):
    s = np.linspace(0, 0.3, 500)
    
    # 1. Polynomial weight function gamma(s)
    cutoff = m * kappa_epsilon
    gamma = np.where(s <= cutoff, 
                     (1 - s/cutoff)**(2*m) * (2*s/kappa_epsilon + 1), 
                     0)
    
    # 2. Exponential function
    exp_curve = np.exp(-(s**a) / (h**a))

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(s, gamma, 'r-', label=f'Gamma Curve (m={m}, κε={kappa_epsilon})')
    plt.plot(s, exp_curve, 'b-', label=f'Gaussian Curve (h={h}, a={a})')
    
    plt.ylim(0, 1.1)
    plt.xlim(0, 0.3)
    plt.xlabel('s')
    plt.ylabel('Weight')
    plt.title('Comparison of Weight Functions')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    folder = "./plots"
    filename = f"weight_plot_m{m}_ke{kappa_epsilon}_h{h}_a{a}.png"
    filepath = os.path.join(folder, filename)
    
    # Save and close
    plt.savefig(filepath, dpi=300)
    plt.close() # Close plot to free up memory if running in a loop
    print(f"Saved: {filepath}")
    plt.show()

# Example: Generate the plot with parameters from Figure 1
plot_weight_functions(m=10, kappa_epsilon=0.08, h=0.2, a=10)