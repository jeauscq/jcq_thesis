import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Load CSV (assumes columns include 'step' and 'value')
df = pd.read_csv("/home/jeauscq/Desktop/loss_vs_steps_8M.csv")

plt.figure(figsize=(8, 5))
plt.plot(df['Step'], df['Value'], label='Discriminator Loss', color='darkblue')
plt.xlabel("Step")
plt.ylabel("Validation Loss")
plt.title("Validation Loss over Steps")
plt.grid(False)

# Make y-axis ticks more granular
plt.gca().yaxis.set_major_locator(MultipleLocator(0.75))  # Set to 0.1 or 0.05 if needed

# plt.legend()
plt.tight_layout()
plt.savefig("loss_plot.pdf")
plt.show()