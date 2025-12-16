#!/usr/bin/env python3
import argparse, os, subprocess, json, shutil, tarfile, time
from pathlib import Path
import yaml
from jinja2 import Environment, FileSystemLoader

HOME = Path(os.environ["HOME"]).resolve()
STORE = Path(os.environ["STORE"]).resolve()
LUSTRE = Path(os.environ["LUSTRE"]).resolve()

def run(cmd, cwd=None, capture=False):
    p = subprocess.run(cmd, cwd=cwd, shell=True, check=True,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.STDOUT if capture else None,
                       text=True)
    return p.stdout if capture else ""

def snapshot_code(src_dir: Path, out_tar: Path):
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, "w:gz") as tar:
        for rel in ["exps", "conf"]:
            base = src_dir / rel
            if base.exists():
                tar.add(base, arcname=rel)

def compose_job_id(project, cfg, exp_path=None):
    # Add seconds to timestamp (was "%m%d.%H%M", now "%m%d.%H%M%S")
    stamp = time.strftime("%m%d.%H%M%S")
    cpus = cfg["slurm"]["cpus"]
    mem = cfg["slurm"]["mem_per_cpu"]
    gpus = cfg["slurm"]["gpus"]
    exp_suffix = f"__{exp_path.replace('/', '_')}" if exp_path else ""

    # Parse slurm_time and build "dd-hhmmss" with real values, all zero padded
    slurm_time = cfg["slurm"]["time"]
    if '-' in slurm_time:
        days, hms = slurm_time.split('-')
    else:
        days = "0"
        hms = slurm_time

    # Split hours, mins, secs, pad with zeros as needed
    time_parts = hms.strip().split(':')
    # SLURM allows H, H:M, H:M:S
    hours = int(time_parts[0]) if len(time_parts) > 0 and time_parts[0] else 0
    mins = int(time_parts[1]) if len(time_parts) > 1 and time_parts[1] else 0

    ddhhmm = f"{int(days):02d}-{hours:02d}{mins:02d}"
    return f"{project}{exp_suffix}__{stamp}__{ddhhmm}_{cpus}c_{mem}m_{gpus}g"

def main():
    ap = argparse.ArgumentParser(description="FT3 orchestrator: prepares the run and submits sbatch.")
    ap.add_argument("--path", required=True, help="Full path: project/exp/ver (e.g., myproject/exp1/v1)")
    ap.add_argument("--conf", default="conf/job.yaml", help="Main YAML (ignored if path has exp/ver)")
    ap.add_argument("--submit", action="store_true", help="Submit with sbatch after generation")
    ap.add_argument("--array", default=None, help="Array expression e.g. 0-9")
    ap.add_argument("--comment", default="", help="Comment to add to the experiment run")
    args = ap.parse_args()

    # Parse the path: project/exp/ver
    path_parts = args.path.split('/')
    project = path_parts[0]
    exp_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else None
    
    proj_dir = HOME / "projects" / project
    assert proj_dir.exists(), f"Project directory does not exist: {proj_dir}"

    # 1) Load configs - experiment path is required
    assert exp_path, "Experiment path is required. Use format: project/exp or project/exp/version"
    
    exp_dir = proj_dir / "exps" / exp_path
    assert exp_dir.exists(), f"Experiment directory does not exist: {exp_dir}"
    
    job_yaml = exp_dir / "conf" / "job.yaml"
    input_yaml = exp_dir / "conf" / "input_params.yaml"
    
    assert job_yaml.exists(), f"job.yaml not found: {job_yaml}"
    assert input_yaml.exists(), f"input_params.yaml not found: {input_yaml}"
    
    with open(job_yaml) as f:
        job_cfg = yaml.safe_load(f)
    with open(input_yaml) as f:
        input_cfg = yaml.safe_load(f)

    # Normalize input parameters to always live under "input_params"
    # Support both legacy nested format and new root-level format
    if isinstance(input_cfg, dict) and "input_params" in input_cfg and isinstance(input_cfg["input_params"], dict):
        normalized_input_params = input_cfg["input_params"]
    elif isinstance(input_cfg, dict):
        normalized_input_params = input_cfg
    else:
        normalized_input_params = {}

    # Compose unified cfg with normalized input_params
    cfg = {**job_cfg, "input_params": normalized_input_params}
    code_dir = exp_dir / "src"

    # 2) No overrides (removed)

    user = os.environ.get("USER", "user")
    job_id = compose_job_id(project, cfg, exp_path)

    # 3) Runtime paths - use unified project/exp/ver structure
    if exp_path:
        # Use the exp path directly for directory structure (keep / for directories)
        run_dir = LUSTRE / "projects" / project / "runs" / exp_path / job_id
        store_artifacts = STORE / "projects" / project / "artifacts" / exp_path / job_id
    else:
        # Legacy mode - no exp specified
        run_dir = LUSTRE / "projects" / project / "runs" / job_id
        store_artifacts = STORE / "projects" / project / "artifacts" / job_id
    
    paths = {
        "work": str(run_dir / "work"),
        "out": str(run_dir / "out"),
        "env": str(run_dir / "env"),
        "venv": str(STORE / "projects" / project / "venv"),
        "code": str(code_dir),
        "store_artifacts": str(store_artifacts),
        "store_data": str(STORE / "projects" / project / "data"),
    }
    for p in [paths["work"], paths["out"], paths["env"], paths["store_artifacts"], paths["store_data"]]:
        Path(p).mkdir(parents=True, exist_ok=True)

    # 4) Link data (in -> STORE/data)
    in_dir = run_dir / "in"
    if in_dir.exists() or in_dir.is_symlink():
        try: in_dir.unlink()
        except: shutil.rmtree(in_dir, ignore_errors=True)
    in_dir.symlink_to(Path(paths["store_data"]), target_is_directory=True)

    # 5) resolved.yaml
    resolved = {
        "title": cfg["title"], 
        "description": cfg["description"],
        "input_params": cfg["input_params"],
        "slurm": cfg["slurm"],
        "notify": cfg.get("notify", {"email": f"{user}@example.com"}),
        "paths": paths, 
        "job_id": job_id,
        "project": project,
        "exp_path": exp_path or "",
        "full_path": args.path,
        "input_params_path": str(input_yaml),
        "comment": args.comment,
    }
    Path(paths["work"]).mkdir(parents=True, exist_ok=True)
    with open(Path(paths["work"]) / "resolved.yaml", "w") as f:
        yaml.safe_dump(resolved, f)

    # 6) meta + snapshot
    meta = {
        "job_id": job_id, 
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project": project,
        "exp_path": exp_path or "",
        "full_path": args.path
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    snapshot_code(proj_dir, run_dir / "code.tar.gz")

    # 7) Render run.sh from template
    env = Environment(loader=FileSystemLoader(str(proj_dir / "conf")))
    run_sh = env.get_template("run.sh.j2").render(**resolved)
    with open(run_dir / "run.sh", "w") as f:
        f.write(run_sh)
    os.chmod(run_dir / "run.sh", 0o755)

    print(f"[OK] Generated: {run_dir}")

    # 8) Queue submission
    if args.submit:
        sb = f"sbatch {'--array='+args.array if args.array else ''} {run_dir}/run.sh"
        print(f"[SUBMIT] {sb}")
        out = run(sb, capture=True)
        print(out)

if __name__ == "__main__":
    main()
