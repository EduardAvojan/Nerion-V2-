import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("telemetry.csv")

df.plot.scatter(x="surprise", y="dynamic_lr", title="Surprise vs. Dynamic Learning Rate")
plt.xlabel("Surprise")
plt.ylabel("Dynamic Learning Rate")
plt.grid(True)
plt.savefig("surprise_vs_lr.png")

print("Successfully generated surprise_vs_lr.png")