"""
Data Collection Entry Point for Unitree G1 with Dex3 Hands.
Uses the provided YAML configuration to initialize the recording session.
"""

import sys
import os

# 1. Register custom G1 modules into the global registry
try:
    import robot.unitree_g1
    import teleop.unitree_g1
except ImportError as e:
    print(f"\n[IMPORT ERROR]: Failed to load custom G1 modules: {e}")
    sys.exit(1)

# Import the core recording logic from the LeRobot library
from lerobot.scripts.lerobot_record import main

def display_help():
    """Prints a professional usage guide for data collection."""
    print("\n" + "="*70)
    print("LEROBOT DATA COLLECTION INTERFACE - UNITREE G1")
    print("="*70)
    print("\nUSAGE:")
    print("  python init_lerobot_record.py --config_path=<PATH_TO_YAML_CONFIG> [OPTIONS]")
    print("\nARGUMENTS:")
    print("  --config_path     Path to the recording YAML configuration file.")
    print("  --sim             Short flag to force simulation mode.")
    print("  --simulation=true Force simulation mode.")
    print("  -h, --help        Show this help message and exit.")
    
    # --- LISTAGEM DINÂMICA DOS ARQUIVOS YAML ---
    print("\nAVAILABLE CONFIG FILES:")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, "config")
    
    if os.path.exists(config_dir) and os.path.isdir(config_dir):
        # Filtra apenas os arquivos .yaml ou .yml
        yaml_files = [f for f in os.listdir(config_dir) if f.endswith(('.yaml', '.yml'))]
        if yaml_files:
            for yml in sorted(yaml_files):
                print(f"  - config/{yml}")
        else:
            print("  (No .yaml files found in 'config/' folder)")
    else:
        print("  (Folder 'config/' not found)")
    # -------------------------------------------

    print("\nEXAMPLE:")
    print("  python init_lerobot_record.py --config_path=config/record_g1.yaml --sim\n")
    print("="*70 + "\n")

if __name__ == "__main__":
    cli_args = sys.argv[:]
    
    help_flags = ["-h", "--help", "-help"]
    if any(flag in cli_args for flag in help_flags):
        display_help()
        sys.exit(0)

    has_config_path = any("--config_path" in arg for arg in cli_args)
    if not has_config_path:
        print("\n[CRITICAL ERROR]: Mandatory '--config_path' argument is missing.")
        print("Use '-h' or '--help' for usage instructions.")
        sys.exit(1)

    force_sim = False
    for arg in cli_args:
        if arg in ["--sim", "--simulation=true"]:
            force_sim = True
            if arg in sys.argv:
                sys.argv.remove(arg)

    if force_sim:
        sys.argv.append("--robot.is_simulation=true")
        sys.argv.append("--teleop.is_simulation=true") 
        print("[INFO]: Overriding YAML config: robot and teleop is_simulation set to TRUE")
    else:
        sys.argv.append("--robot.is_simulation=false")
        sys.argv.append("--teleop.is_simulation=false") 
        print("[INFO]: Using Real Robot mode (robot and teleop is_simulation=false)")

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[SYSTEM]: Data collection session terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[RUNTIME ERROR]: {e}")
        sys.exit(1)