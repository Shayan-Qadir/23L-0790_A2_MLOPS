import os
import shutil
import subprocess
from pathlib import Path

def run_cmd(cmd, cwd=None):
    """Run shell command and return stdout/stderr."""
    print(f"\n[RUNNING]: {cmd}")
    result = subprocess.run(cmd, cwd=cwd, shell=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(f"[STDERR]: {result.stderr.strip()}")
    return result

def main():
    root_dir = Path("g:/KAAM/UNI/MLOPS/A2/23L0790_A2_Ops")
    toy_dir = root_dir / "toy_dvc_lab"
    remote_dir = root_dir / "toy_dvc_remote"
    
    print("=" * 70)
    print(" TASK 1.4: DVC TOY DEMO EXPERIMENT")
    print("=" * 70)
    
    # Clean previous demo runs if any
    if toy_dir.exists():
        shutil.rmtree(toy_dir, ignore_errors=True)
    if remote_dir.exists():
        shutil.rmtree(remote_dir, ignore_errors=True)
        
    toy_dir.mkdir(parents=True, exist_ok=True)
    remote_dir.mkdir(parents=True, exist_ok=True)
    
    dvc_bin = str(root_dir / ".venv" / "Scripts" / "dvc.exe")
    
    # 1. Initialize Git & DVC
    run_cmd("git init", cwd=toy_dir)
    run_cmd(f'"{dvc_bin}" init', cwd=toy_dir)
    run_cmd('git commit -m "Initialize Git and DVC"', cwd=toy_dir)
    
    # 2. Create small CSV (Version 1)
    csv_file = toy_dir / "sample_data.csv"
    with open(csv_file, "w") as f:
        f.write("id,label,score\n1,real,0.92\n2,ai,0.88\n3,real,0.95\n")
    print(f"\n[+] Created Version 1 CSV (3 records)")
    
    # 3. Track with DVC
    run_cmd(f'"{dvc_bin}" add sample_data.csv', cwd=toy_dir)
    run_cmd('git add sample_data.csv.dvc .gitignore', cwd=toy_dir)
    run_cmd('git commit -m "DVC: Track sample_data.csv v1"', cwd=toy_dir)
    
    # 4. Configure Remote Storage
    run_cmd(f'"{dvc_bin}" remote add -d toy_remote "{remote_dir}"', cwd=toy_dir)
    run_cmd('git add .dvc/config', cwd=toy_dir)
    run_cmd('git commit -m "Configure local directory DVC remote storage"', cwd=toy_dir)
    
    # 5. Push to Remote
    run_cmd(f'"{dvc_bin}" push', cwd=toy_dir)
    
    # 6. Modify data (Version 2)
    with open(csv_file, "a") as f:
        f.write("4,ai,0.76\n5,ai,0.84\n6,real,0.99\n")
    print(f"\n[+] Updated to Version 2 CSV (6 records)")
    
    # 7. Track Version 2
    run_cmd(f'"{dvc_bin}" add sample_data.csv', cwd=toy_dir)
    run_cmd('git commit -am "DVC: Update sample_data.csv to v2"', cwd=toy_dir)
    run_cmd(f'"{dvc_bin}" push', cwd=toy_dir)
    
    # 8. Return to older version (v1)
    print("\n" + "=" * 70)
    print(" ROLLING BACK TO VERSION 1")
    print("=" * 70)
    run_cmd("git checkout HEAD~1 sample_data.csv.dvc", cwd=toy_dir)
    run_cmd(f'"{dvc_bin}" checkout', cwd=toy_dir)
    with open(csv_file) as f:
        v1_content = f.read()
    print("[CONTENT OF sample_data.csv AFTER CHECKOUT TO v1]:\n" + v1_content.strip())
    
    # 9. Return to Version 2
    print("\n" + "=" * 70)
    print(" RETURNING TO VERSION 2")
    print("=" * 70)
    run_cmd("git checkout HEAD sample_data.csv.dvc", cwd=toy_dir)
    run_cmd(f'"{dvc_bin}" checkout', cwd=toy_dir)
    with open(csv_file) as f:
        v2_content = f.read()
    print("[CONTENT OF sample_data.csv AFTER CHECKOUT TO v2]:\n" + v2_content.strip())
    
    # 10. Delete local data & Pull from remote
    print("\n" + "=" * 70)
    print(" TESTING REMOTE PUSH & PULL RECOVERY")
    print("=" * 70)
    os.remove(csv_file)
    print(f"[!] Deleted local file: {csv_file.name} (Exists: {csv_file.exists()})")
    run_cmd(f'"{dvc_bin}" pull', cwd=toy_dir)
    print(f"[+] Restored from remote: {csv_file.name} (Exists: {csv_file.exists()})")
    with open(csv_file) as f:
        restored_content = f.read()
    print("[RESTORED CONTENT]:\n" + restored_content.strip())
    
    print("\n" + "=" * 70)
    print(" TOY DVC DEMO COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()

