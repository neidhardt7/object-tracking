import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the Excel file for both sheets
df1 = pd.read_excel("sep.xlsx", sheet_name="Relative_Separation")#Displacement_Time")
#df2 = pd.read_excel("dt2.xlsx", sheet_name="Sheet2")

# Get the column names
col2_name = "Time"  # Second column name
col3_name = "Separation(in mm)"  # Third column name

# Extract the data from first sheet
x1 = df1.iloc[:, 0]  # Second column 1
y1 = df1.iloc[:, 1]  # Third column 2

# Extract the data from second sheet
#x2 = df2.iloc[:, 1]  # Second column
#y2 = df2.iloc[:, 2]  # Third column

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x1, y1, 'bo-', label='Drop1 Data')  # Blue dots with lines
#plt.plot(x2, y2, 'ro-', label='Drop2 Data')  # Red dots with lines

# Add labels and title
plt.xlabel(col2_name)
plt.ylabel(col3_name)
plt.title(f'{col3_name} vs {col2_name}')

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend()

# Customize the appearance
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()

# Save the plot
plt.savefig('separation_graph.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Print some basic statistics for both datasets
print("\nData Statistics:")
print("\nSheet1 Statistics:")
print(f"Number of data points: {len(x1)}")
print(f"\n{col2_name} statistics:")
print(f"Mean: {x1.mean():.2f}")
print(f"Std Dev: {x1.std():.2f}")
print(f"\n{col3_name} statistics:")
print(f"Mean: {y1.mean():.2f}")
print(f"Std Dev: {y1.std():.2f}")

##print("\nSheet2 Statistics:")
##print(f"Number of data points: {len(x2)}")
##print(f"\n{col2_name} statistics:")
##print(f"Mean: {x2.mean():.2f}")
##print(f"Std Dev: {x2.std():.2f}")
##print(f"\n{col3_name} statistics:")
##print(f"Mean: {y2.mean():.2f}")
##print(f"Std Dev: {y2.std():.2f}") 
