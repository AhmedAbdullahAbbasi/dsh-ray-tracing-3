"""Install the optional driver hook while preserving science configuration.

Run from the extracted package with --repo PATH. Git author configuration is
not needed. This does not create a commit or push anything.
"""
import argparse
import ast
import shutil
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,required=True)
    args=parser.parse_args();repo=args.repo.resolve();source=Path(__file__).resolve().parent
    if not (repo/"dsh/physics/newdust.py").is_file() or not (repo/"run_simulation.py").is_file():
        raise ValueError("expected the development/physics-improvements repository layout")
    patch=source/"driver_hook.patch"
    command=["git","apply","--include=run_simulation.py"]
    check=subprocess.run(command+["--check",str(patch)],cwd=repo,capture_output=True,text=True)
    already=False
    if check.returncode:
        reverse=subprocess.run(command+["--reverse","--check",str(patch)],cwd=repo,capture_output=True,text=True)
        if reverse.returncode:
            raise RuntimeError("driver differs from the tested patch; review driver_hook.patch before applying:\n"+check.stderr)
        already=True
    target=repo/"reflex_adapter"
    if source!=target:
        shutil.copytree(source,target,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__"))
    if not already:
        subprocess.run(command+[str(target/"driver_hook.patch")],cwd=repo,check=True)
    config=repo/"sim_config.py"
    assignments=set()
    for node in ast.parse(config.read_text()).body:
        if isinstance(node,ast.Assign):
            assignments.update(t.id for t in node.targets if isinstance(t,ast.Name))
    defaults={"MATERIAL_MODEL":'"default"',"REFLEX_MATERIAL_DIRECTORY":'"reflex_adapter/materials"',
              "REFLEX_ABSORPTION_ENABLED":"True"}
    missing=[f"{name} = {value}" for name,value in defaults.items() if name not in assignments]
    if missing:
        with config.open("a",encoding="utf-8",newline="\n") as stream:
            stream.write("\n\n# Optional RefleX material comparison; existing science settings are preserved.\n")
            stream.write("\n".join(missing)+"\n")
    print("RefleX driver hook ready. Existing material remains the default.")
    print("Verify with: python reflex_adapter/verify_materials.py --repository .")


if __name__=="__main__":main()
