import re

# Read main.py
with open('main.py', 'r') as f:
    main_code = f.read()

# Read config.json
import json
with open('config.json', 'r') as f:
    config = json.load(f)

# Extract all config assignments in main.py
assignments = re.findall(r'(\w+)\s*=\s*\w+_config\["([^"]+)"\]', main_code)
assignments_get = re.findall(r'(?:_|[a-z_]+)_config\.get\("([^"]+)"', main_code)
assignments_get += re.findall(r'config\.get\("([^"]+)"', main_code)

used_config_keys = set()
for var_name, key in assignments:
    used_config_keys.add(key)
    
for key in assignments_get:
    used_config_keys.add(key)

# Check which config keys from a2c_exploration are not used
a2c_keys = set(config.get("a2c_exploration", {}).keys())
used_a2c = set()
for var_name, key in assignments:
    if key in a2c_keys:
        used_a2c.add(key)
for key in assignments_get:
    if key in a2c_keys:
        used_a2c.add(key)

unused_a2c = a2c_keys - used_a2c
if unused_a2c:
    print("Unused a2c_exploration config keys:", unused_a2c)

# Check plotting config
plotting_keys = set(config.get("plotting", {}).keys())
used_plotting = set()
for var_name, key in assignments:
    if key in plotting_keys:
        used_plotting.add(key)
for key in assignments_get:
    if key in plotting_keys:
        used_plotting.add(key)

unused_plotting = plotting_keys - used_plotting
if unused_plotting:
    print("Unused plotting config keys:", unused_plotting)

print("\nAll used config keys:", used_config_keys)
