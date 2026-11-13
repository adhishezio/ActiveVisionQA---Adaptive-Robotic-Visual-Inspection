from pathlib import Path
from setuptools import find_packages, setup

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/activevisionqa"]),
    ("share/activevisionqa", ["package.xml"]),
]
for folder in ("launch", "config", "urdf"):
    for path in Path(folder).rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            data_files.append((str(Path("share/activevisionqa") / path.parent), [str(path)]))

setup(
    name="activevisionqa", version="0.1.0", packages=find_packages(exclude=["tests"]),
    data_files=data_files, install_requires=["setuptools"], zip_safe=False,
    maintainer="ActiveVisionQA contributors", maintainer_email="maintainer@example.invalid",
    description="Reproducible simulation-only active visual inspection",
    license="LicenseRef-Unspecified",
    entry_points={"console_scripts": [
        "inspect = activevisionqa.cli:main",
        "world_metadata = activevisionqa.world_metadata:main",
    ]},
)
