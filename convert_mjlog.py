import os
import click
import subprocess
from pathlib import Path
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
from collections import deque


def convert(mp):

    mjlog_dir, mjson_path = mp
    relative = mjson_path.relative_to(mjlog_dir)
    input_host = mjlog_dir
    input_container = Path("/input")
    input_path = input_container / \
        relative.parent / \
        mjson_path.name

    output_host_base = Path("./output")
    output_container = Path("/output")
    output_relative = \
        relative.parent / \
        f"{mjson_path.stem}.mjson"
    output_path = output_container / output_relative
    output_host = output_host_base / output_relative

    output_host.parent.mkdir(parents=True, exist_ok=True)

    command = f"docker run --rm -v {input_host.absolute()}:/input "\
        + f" -v {output_host_base.absolute()}:{output_container.absolute()} manue:dev " \
        + f" /bin/bash convert.sh {input_path.absolute()} {output_path.absolute()}"  # + f" mjai convert {input_path.absolute()} {output_path.absolute()}"
    proc = subprocess.Popen(command, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    proc.wait()

    if proc.stdout.readline():
        # have error message
        return 1
    else:
        return 0


@click.command()
@click.option('--mjlog_dir', type=str)
def main(mjlog_dir):
    mjlog_dir = Path(mjlog_dir)
    # mjson_paths = Path(mjlog_dir).glob("201212/*.mjlog")
    mjson_paths = Path(mjlog_dir).glob("**/*.mjlog")
    mjson_paths = sorted([(mjlog_dir, p) for p in mjson_paths])
    mjson_paths = mjson_paths
    print(len(mjson_paths))
    results = deque()
    pool = Pool(processes=multiprocessing.cpu_count())
    with tqdm(total=len(mjson_paths)) as t:
        for result in pool.imap_unordered(convert, mjson_paths):
            results.append(result)
            t.update(1)

    print(
        f"converted:{len([r for r in results if r == 0])} files, error:{len([r for r in results if r == 1])}")


if __name__ == "__main__":
    main()
