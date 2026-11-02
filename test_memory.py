
from memory import save_memory, get_all_memories

# Save personal information
save_memory("name", "Nikhil Singh")
save_memory("branch", "CSIT")

# Save multiple interests
save_memory(
    "interests",
    "Backend Development, Machine Learning, DSA, Python"
)

# Retrieve all memories
memories = get_all_memories()

print("\nSaved User Profile")
print("-" * 30)

for key, value in memories:
    print(f"{key}: {value}")
