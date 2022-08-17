r"""Darken Python code exposed to sunlight"""

import black
import click
import concurrent.futures as cf
import difflib
import git
import inspect
import re

from multiprocessing import Lock, Manager
from pathlib import Path
from typing import *


def parse_range(start: str, length: str) -> range:
    r"""Parses hunk range."""

    start = int(start)
    length = int(length) if length else 1

    return range(start, start + length)


def git_diff(commit: str, repository: Path = None) -> Dict[Path, Set[int]]:
    r"""Compares the working tree to a commit and returns the lines that have changed."""

    commit = git.Repo(repository).commit(commit)
    lines = {}

    for diff in commit.diff(None, create_patch=True, unified=0):
        if diff.deleted_file:
            continue

        file = Path(diff.b_path)
        diff = diff.diff.decode('utf8')

        if diff:
            ranges = [
                parse_range(s, l)
                for s, l in re.findall(r'@@ .* \+(\d+),?(\d+)? @@', diff)
            ]

            lines[file] = set().union(*ranges)

    return lines


def format_lines(
    src: str,
    lines: Set[int],
    fast: bool,
    mode: black.Mode,
) -> str:
    r"""Formats source code with Black but only keeps changes that overlap lines.

    Returns the reformatted code. If `fast` is `False`, confirms that the
    reformatted code is equivalent to the source code.
    """

    # Format with black
    dst = black.format_str(src, mode=mode)

    if src == dst:
        raise black.report.NothingChanged

    # Get unified diff hunks
    src_ = src.splitlines(keepends=True)
    dst_ = dst.splitlines(keepends=True)

    hunks = []

    for line in difflib.unified_diff(src_, dst_, n=0):
        if line.startswith('---') or line.startswith('+++'):
            pass
        elif line.startswith('@@'):
            hunks.append([line])
        else:
            hunks[-1].append(line)

    # Filter hunks with respect to lines
    patch = []

    for hunk in hunks:
        s, l = re.match(r'@@ -(\d+),?(\d+)? .* @@', hunk[0]).groups()
        rnge = parse_range(s, l)

        if lines.intersection(rnge):
            patch.extend(hunk)

    if not patch:
        raise black.report.NothingChanged

    # Apply (filtered) patch
    i, dst_ = 0, []

    for line in patch:
        if line.startswith('@@'):
            j = int(re.match(r'@@ -(\d+),?(\d+)? .* @@', line).group(1)) - 1
            dst_.extend(src_[i:j])
            i = j
        elif line.startswith(' '):
            dst_.append(src_[i])
            i += 1
        elif line.startswith('+'):
            dst_.append(line[1:])
        elif line.startswith('-'):
            i += 1

    dst_.extend(src_[i:])
    dst = ''.join(dst_)

    # Assert equivalence
    if not fast:
        black.assert_equivalent(src, dst)

    return dst


def format_file(
    file: Path,
    lines: Set[int],
    fast: bool,
    mode: black.Mode,
    check: bool = False,
    diff: bool = False,
    color: bool = False,
    lock: Optional[Lock] = None,
) -> bool:
    r"""Formats a file with Black but only keeps changes that overlap lines.

    Returns whether the file changes. If `diff` is `True`, writes a unified
    diff to standard output. If `diff` and `check` are `False`, writes
    reformatted code to the file.
    """

    with open(file, 'rb') as f:
        src, encoding, newline = black.decode_bytes(f.read())

    try:
        dst = format_lines(
            src,
            lines,
            fast=fast,
            mode=mode,
        )
    except black.report.NothingChanged:
        return False

    if check:
        pass
    elif diff:
        diff = black.diff(src, dst, str(file), str(file))

        if color:
            diff = black.color_diff(diff)

        with lock or black.nullcontext():
            click.echo(diff)
    else:
        with open(file, 'w', encoding=encoding, newline=newline) as f:
            f.write(dst)

    return True


def borrow(command: click.Command) -> Callable:
    r"""Returns a decorator that copies the parameters of a click command to
    another function."""

    def wrapper(f: Callable) -> Callable:
        signature = inspect.signature(f)

        for p in reversed(command.params):
            if p.name in signature.parameters:
                click.decorators._param_memo(f, p)

        return f

    return wrapper

click.borrow = borrow


@click.command()
@click.borrow(black.main)
@click.option(
    '--commit',
    default='HEAD',
    help="The commit to which the working tree is compared.",
    show_default=True,
)
@click.pass_context
def tan(
    ctx: click.Context,
    line_length: int,
    target_version: List[black.TargetVersion],
    skip_string_normalization: bool,
    skip_magic_trailing_comma: bool,
    preview: bool,
    check: bool,
    diff: bool,
    color: bool,
    fast: bool,
    quiet: bool,
    verbose: bool,
    workers: int,
    include: Optional[re.Pattern],
    exclude: Optional[re.Pattern],
    extend_exclude: Optional[re.Pattern],
    force_exclude: Optional[re.Pattern],
    src: Tuple[str, ...],
    config: Optional[str],
    commit: Optional[str],
) -> None:
    r"""Darken Python code exposed to sunlight"""

    if not src:
        src = ('.',)

    ctx.ensure_object(dict)
    ctx.obj['root'], _ = black.find_project_root(src)

    mode = black.Mode(
        target_versions=set(target_version),
        line_length=line_length,
        string_normalization=not skip_string_normalization,
        magic_trailing_comma=not skip_magic_trailing_comma,
        preview=preview,
    )

    report = black.Report(check=check, diff=diff, quiet=quiet, verbose=verbose)

    if force_exclude:
        force_exclude = re.compile(force_exclude.pattern + r'|\.ipynb')
    else:
        force_exclude = re.compile(r'\.ipynb')

    sources = black.get_sources(
        ctx=ctx,
        src=src,
        quiet=quiet,
        verbose=verbose,
        include=include,
        exclude=exclude,
        extend_exclude=extend_exclude,
        force_exclude=force_exclude,
        report=report,
        stdin_filename=None,
    )

    lines = git_diff(commit, ctx.obj['root'])
    sources = sources.intersection(lines.keys())

    if not sources:
        if verbose or not quiet:
            click.echo(f"No Python files have changed from {commit}.", err=True)
        ctx.exit(0)

    try:
        executor = cf.ProcessPoolExecutor(max_workers=min(workers, len(sources)))
    except:
        executor = cf.ThreadPoolExecutor(max_workers=1)

    with executor:
        lock = Manager().Lock()
        tasks = {}

        for file in sources:
            tasks[file] = executor.submit(
                format_file,
                file,
                lines[file],
                fast=fast,
                mode=mode,
                check=check,
                diff=diff,
                color=color,
                lock=lock,
            )

    for file, task in tasks.items():
        if task.exception():
            report.failed(file, str(task.exception()))
        else:
            if task.result():
                report.done(file, black.Changed.YES)
            else:
                report.done(file, black.Changed.NO)

    if verbose or not quiet:
        if verbose or report.change_count or report.failure_count:
            click.echo(err=True)
        click.echo(str(report), err=True)

    ctx.exit(report.return_code)
