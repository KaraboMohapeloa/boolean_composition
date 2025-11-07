import deepdish as dd

# Load data from softmax directory
source_path = 'exps_data/softmax/prime_experiment_stats.h5'
data = dd.io.load(source_path)

# Save as tested.h5 in vanilla directory
dest_path = 'exps_data/vanilla/tested.h5'
dd.io.save(dest_path, data)

print(f"Successfully copied data from:")
print(f"  {source_path}")
print(f"To:")
print(f"  {dest_path}")
