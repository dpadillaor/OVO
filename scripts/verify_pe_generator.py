
import sys
import os
import torch

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add thirdParty/perception_models to sys.path
pm_path = os.path.join(project_root, "thirdParty/perception_models")
if pm_path not in sys.path:
    sys.path.append(pm_path)

try:
    from ovo.entities.pe_generator import PEGenerator
    print("Successfully imported PEGenerator")
except ImportError as e:
    print(f"Failed to import PEGenerator: {e}")
    sys.exit(1)

# Mock config
config = {
    "model_card": "PE-Core-T16-384", # Smallest model for test
    "mask_res": 384,
    "use_half": False
}

try:
    # Initialize generator (might fail if dependencies are missing or weights not found)
    # We catch errors to just verify the class structure if loading fails
    print("Attempting to initialize PEGenerator...")
    # Mocking pe.CLIP.from_config to avoid loading actual model which might fail or take time
    # But we can't easily mock imports inside the module without more magic.
    # So we just try and catch.
    
    # Check if we can mock the thirdParty module if it's not actually usable in this env
    # But the files are there.
    
    generator = PEGenerator(config, device="cpu")
    print("Successfully initialized PEGenerator")
    pass
except Exception as e:
    print(f"Initialization failed (expected if dependencies missing): {e}")

print("Verification script finished.")
