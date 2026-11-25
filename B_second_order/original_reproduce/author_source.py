"""Build the author's libsvm_cd with a common, measurement-only instrumentation."""

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

from experiment_io import download, read_json, sha256

SOURCE_URL = "https://github.com/albertotb/libsvm_cd.git"
SOURCE_FILES = ("svm.cpp", "svm.h", "svm-train.c", "svm-predict.c", "COPYRIGHT")
ZIG_RELEASE = "zig-x86_64-windows-0.15.2"
ZIG_URL = f"https://ziglang.org/download/0.15.2/{ZIG_RELEASE}.zip"
ZIG_SHA256 = "3a0ed1e8799a2f8ce2a6e6290a9ff22e6906f8227865911fb7ddedc3cc14cb0c"
STOP_CHECK = re.compile(r"(?m)^([ \t]*)if\s*\(\s*Gmax\s*\+\s*Gmax2\s*(<=?)\s*eps\s*\)\s*return\s+1\s*;")
RHO = re.compile(r"(?m)^([ \t]*)si->rho\s*=\s*calculate_rho\(\)\s*;")
OBJECTIVE = re.compile(r"(?m)^([ \t]*)si->obj\s*=\s*[^;\n]+;")
TRAIN_CALL = re.compile(r"(?m)^([ \t]*)model\s*=\s*svm_train\s*\(\s*&prob\s*,\s*&param\s*\)\s*;")

SOLVER_HEADER = """\
#include <limits>
double group19_gmax = std::numeric_limits<double>::quiet_NaN();
double group19_gmax2 = std::numeric_limits<double>::quiet_NaN();
double group19_obj = std::numeric_limits<double>::quiet_NaN();
int group19_kkt_source = 0;
"""

TRAIN_HEADER = """\
#include <chrono>
extern double group19_gmax, group19_gmax2, group19_obj;
extern int group19_kkt_source;
"""


def indent(code, whitespace):
    return "\n".join(whitespace + line if line else "" for line in code.strip("\n").splitlines())


def instrument_solver(source):
    def stop(match):
        return indent("""
if (Gmax + Gmax2 OP eps)
{
    if (active_size == l)
    {
        group19_gmax = Gmax;
        group19_gmax2 = Gmax2;
        group19_kkt_source = 1;
    }
    return 1;
}
""".replace("OP", match[2]), match[1])

    def endpoint(match):
        capture = indent("""
// A run capped at max_iter may not have passed the normal stopping check.
if (group19_kkt_source == 0 && active_size == l)
{
    group19_gmax = -INF;
    group19_gmax2 = -INF;
    for (int t = 0; t < l; ++t)
    {
        const bool positive = y[t] == +1;
        if (positive ? !is_upper_bound(t) : !is_lower_bound(t))
            group19_gmax = max(group19_gmax, -y[t] * G[t]);
        if (positive ? !is_lower_bound(t) : !is_upper_bound(t))
            group19_gmax2 = max(group19_gmax2, y[t] * G[t]);
    }
    group19_kkt_source = 2;
}
""", match[1])
        return capture + "\n" + match[0]

    source, n_stop = STOP_CHECK.subn(stop, source)
    source, n_rho = RHO.subn(endpoint, source)
    source, n_objective = OBJECTIVE.subn(lambda m: m[0] + "\n" + m[1] + "group19_obj = si->obj;", source)
    if min(n_stop, n_rho, n_objective) < 1:
        raise ValueError("The author source does not match the expected solver instrumentation points.")
    return SOLVER_HEADER + source, {"stop_checks": n_stop, "endpoint_checks": n_rho, "objectives": n_objective}


def instrument_training(source):
    replacement = r"""
const auto group19_start = std::chrono::steady_clock::now();
model = svm_train(&prob,&param);
const double group19_time =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - group19_start).count();
printf("GROUP19_END %.17g %.17g %.17g %.17g %d\n",
       group19_time, group19_gmax, group19_gmax2, group19_obj, group19_kkt_source);

const char *group19_alpha_path = getenv("GROUP19_ALPHA_PATH");
if (group19_alpha_path)
{
    if (model->param.svm_type != C_SVC || model->nr_class != 2 || !model->sv_indices)
    {
        fprintf(stderr, "GROUP19_ALPHA_PATH requires a binary C-SVC model\n");
        return 2;
    }
    FILE *group19_alpha_file = fopen(group19_alpha_path, "w");
    if (!group19_alpha_file)
    {
        fprintf(stderr, "Cannot open GROUP19_ALPHA_PATH\n");
        return 2;
    }
    for (int k = 0; k < model->l; ++k)
        fprintf(group19_alpha_file, "%d\t%.17g\n", model->sv_indices[k], model->sv_coef[0][k]);
    if (fclose(group19_alpha_file) != 0)
        return 2;
}
"""
    source, count = TRAIN_CALL.subn(lambda m: indent(replacement, m[1]), source)
    if count != 1:
        raise ValueError("Expected one svm_train() call in the author's training CLI.")
    return TRAIN_HEADER + source


def command(args, *, cwd=None, env=None):
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", check=False)
    if result.returncode:
        message = (result.stdout + result.stderr).strip()[-6000:]
        raise RuntimeError(f"{Path(args[0]).name} failed ({result.returncode}): {message}")
    return result.stdout


def choose_compiler(cxx, toolchains):
    if cxx:
        compiler = shutil.which(cxx)
        if compiler is None:
            raise RuntimeError(f"C++ compiler {cxx!r} was not found.")
        return compiler
    for name in ("g++", "clang++", "zig"):
        compiler = shutil.which(name)
        if compiler:
            return compiler
    if os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("A C++ compiler is required. Pass --cxx with the executable path.")

    archive = toolchains / f"{ZIG_RELEASE}.zip"
    compiler = toolchains / ZIG_RELEASE / "zig.exe"
    if not compiler.is_file():
        print("Preparing the portable Zig C++ compiler in work/toolchain …", flush=True)
        download(ZIG_URL, archive, expected_sha256=ZIG_SHA256)
        with ZipFile(archive) as package:
            package.extractall(toolchains)
    if not compiler.is_file():
        raise RuntimeError("Portable Zig archive is missing zig.exe.")
    return str(compiler)


def build(source, destination, cxx, revision):
    compiler = choose_compiler(cxx, destination.parent.parent / "toolchain")
    git = shutil.which("git")
    if not source.exists():
        if git is None:
            raise RuntimeError("Git or an existing author's libsvm_cd checkout is required.")
        source.parent.mkdir(parents=True, exist_ok=True)
        command([git, "clone", SOURCE_URL, str(source)])
        if revision:
            command([git, "-C", str(source), "checkout", "--detach", revision])
    source_sha = {name: sha256(source / name) for name in SOURCE_FILES}
    commit = None
    if git is not None and (source / ".git").exists():
        commit = command([git, "-C", str(source), "rev-parse", "HEAD"]).strip()
        dirty = command([git, "-C", str(source), "diff", "HEAD", "--", *SOURCE_FILES])
        if dirty:
            raise RuntimeError("The supplied author checkout has changed source files.")
    if revision and (commit is None or not commit.startswith(revision)):
        raise RuntimeError("The author's checkout does not match --revision.")

    destination.mkdir(parents=True)
    patched_solver, locations = instrument_solver((source / "svm.cpp").read_text(encoding="utf-8"))
    patched_train = instrument_training((source / "svm-train.c").read_text(encoding="utf-8"))
    (destination / "svm.cpp").write_text(patched_solver, encoding="utf-8")
    (destination / "svm-train.cpp").write_text(patched_train, encoding="utf-8")
    shutil.copyfile(source / "svm-predict.c", destination / "svm-predict.cpp")
    for name in ("svm.h", "COPYRIGHT"):
        shutil.copyfile(source / name, destination / name)
    driver = [compiler]
    build_env = dict(os.environ)
    if Path(compiler).stem.lower() == "zig":
        driver.append("c++")
        if os.name == "nt":
            driver.extend(["-target", "x86_64-windows-gnu"])
        build_env["ZIG_GLOBAL_CACHE_DIR"] = str(destination.parent.parent / "zig-cache")
        build_env["ZIG_LOCAL_CACHE_DIR"] = str(destination / "zig-local-cache")
    flags = ["-O3", "-std=c++14", "-DNDEBUG"]
    suffix = ".exe" if os.name == "nt" else ""
    train = destination / f"svm-train{suffix}"
    predict = destination / f"svm-predict{suffix}"
    command([*driver, *flags, "-c", "svm.cpp", "-o", "svm.o"], cwd=destination, env=build_env)
    for program in (train, predict):
        command([*driver, *flags, f"{program.stem}.cpp", "svm.o", "-o", program.name],
                cwd=destination, env=build_env)
    metadata = {
        "upstream": SOURCE_URL,
        "commit": commit,
        "source_sha256": source_sha,
        "instrumentation_sha256": sha256(Path(__file__)),
        "instrumented_solver_sha256": sha256(destination / "svm.cpp"),
        "instrumented_train_sha256": sha256(destination / "svm-train.cpp"),
        "instrumentation_locations": locations,
        "compiler_version": command([compiler, "--version"]).splitlines()[0],
        "flags": [*driver[1:], *flags],
        "train_binary_sha256": sha256(train),
        "predict_binary_sha256": sha256(predict),
    }
    archive_provenance = source / "archive_provenance.json"
    if archive_provenance.is_file():
        metadata["source_archive"] = read_json(archive_provenance)
    return train, predict, metadata
