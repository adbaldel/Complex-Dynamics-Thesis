import glob
import os
import re

tab = "    "


def parse_tikz_content(file_path):
    """
    Convert a TikZit-generated .tikz file into the body of a .pic.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove wrappers
    content = re.sub(
        r"\\begin{tikzpicture}(?:\[[^\]]*\])?",
        "",
        content,
    )

    content = re.sub(
        r"\\end{tikzpicture}",
        "",
        content,
    )

    # Remove pgfonlayer wrappers
    content = re.sub(
        r"\\begin{pgfonlayer}{[^}]+}",
        "",
        content,
    )

    content = re.sub(
        r"\\end{pgfonlayer}",
        "",
        content,
    )

    # Convert nodes -> coordinates
    content = re.sub(
        r"\\node\s*\[style=none\]\s*"
        r"\(([^)]+)\)\s*"
        r"at\s*"
        r"\(([^)]+)\)\s*"
        r"\{\s*\};",
        r"\\coordinate (\1) at (\2);",
        content,
    )

    # Remove .center references
    content = re.sub(
        r"\(([^)]+)\.center\)",
        r"(\1)",
        content,
    )

    coords = []
    draws = []
    others = []

    for line in content.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith(r"\coordinate"):
            coords.append(line)

        elif line.startswith(r"\draw"):
            draws.append(line)

        else:
            others.append(line)

    blocks = []

    if coords:
        blocks.append(
            "\n".join(
                f"{tab*3}{line}"
                for line in coords
            )
        )

    if draws:
        blocks.append(
            "\n".join(
                f"{tab*3}{line}"
                for line in draws
            )
        )

    if others:
        blocks.append(
            "\n".join(
                f"{tab*3}{line}"
                for line in others
            )
        )

    return "\n\n".join(blocks)


def compile_tikz_file(tikz_path):
    """
    Converts

        blob1.tikz

    into

        blob1.tex

    containing

        \\tikzset{
            blob1/.pic={ ... }
        }
    """

    directory = os.path.dirname(tikz_path)

    name = os.path.splitext(
        os.path.basename(tikz_path)
    )[0]

    output_path = os.path.join(
        directory,
        f"{name}.tex",
    )

    body = parse_tikz_content(tikz_path)

    output = (
        "\\tikzset{\n"
        f"{tab}{name}/.pic={{\n"
        f"{tab*2}\\begin{{scope}}[-]\n"
        f"{body}\n"
        f"{tab*2}\\end{{scope}}\n"
        f"{tab}}}\n"
        "}\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(
        f"[Success] "
        f"{os.path.basename(tikz_path)} "
        f"-> "
        f"{os.path.basename(output_path)}"
    )


def batch_compile_directory(source_dir):

    search_path = os.path.join(
        source_dir,
        "*.tikz"
    )

    tikz_files = sorted(
        glob.glob(search_path)
    )

    if not tikz_files:

        print(
            f"No .tikz files found in "
            f"{source_dir}"
        )

        return

    for tikz_file in tikz_files:

        compile_tikz_file(tikz_file)

    print(
        f"\nCompiled "
        f"{len(tikz_files)} files."
    )


if __name__ == "__main__":

    SOURCE_DIR = "../../figures/tikz/blobs/"

    batch_compile_directory(SOURCE_DIR)