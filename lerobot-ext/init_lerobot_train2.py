"""
Training Entry Point for Unitree G1 with Dex3 Hands (VERSÃO COM TATO).
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import robot.unitree_g1
except ImportError as e:
    print(f"\n[IMPORT ERROR]: Failed to load custom G1 modules: {e}")
    sys.exit(1)

# 3. --- IMPORTA O NOVO MOTOR DE TREINAMENTO ---
try:
    from train.run_train2 import main
except ImportError as e:
    print(f"\n[IMPORT ERROR]: Não encontrei o motor run_train2: {e}")
    sys.exit(1)

def display_help():
    print("\n" + "="*70)
    print("LEROBOT TRAINING INTERFACE - TACTILE ENGINE")
    print("="*70)
    print("\nUSAGE:")
    print("  python init_lerobot_train2.py --config_path=<PATH_TO_YAML_CONFIG> [OPTIONS]")
    print("\nARGUMENTS:")
    print("  --config_path     Path to the training YAML configuration file.")
    print("  -h, --help        Show this help message and exit.")
    
    print("\nAVAILABLE CONFIG FILES:")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, "config", "train")
    
    if os.path.exists(config_dir) and os.path.isdir(config_dir):
        yaml_files = [f for f in os.listdir(config_dir) if f.endswith(('.yaml', '.yml'))]
        if yaml_files:
            for yml in sorted(yaml_files):
                print(f"  - config/train/{yml}")
        else:
            print("  (No .yaml files found in 'config/train/' folder)")
    else:
        print("  (Folder 'config/train/' not found)")

if __name__ == "__main__":
    cli_args = sys.argv[:]
    
    if any(flag in cli_args for flag in ["-h", "--help", "-help"]):
        display_help()
        sys.exit(0)

    has_config_path = any("--config_path" in arg for arg in cli_args)
    if not has_config_path:
        print("\n[CRITICAL ERROR]: Mandatory '--config_path' argument is missing.")
        sys.exit(1)

    print(f"[INFO]: Initializing Custom LeRobot Tactile Training Pipeline...")

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[SYSTEM]: Training session terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[RUNTIME ERROR]: {e}")
        sys.exit(1)