from setuptools import setup, find_packages

# Define your project's metadata
name = "epyfilter"
version = "1.0.0"
description = "Estimate Time-varying Reproduction Number using Ensemble Adjustment Kalman Filter using adaptive inflation"
author = "Han Yong Wunrow"
author_email = "nhw2114@columbia.edu"
url = "https://github.com/hwunrow/rt-estimation"
license = "MIT"

# Define your project's dependencies
install_requires = [
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
]

test_requirements = [
    "pytest",
    "pytest-mock",
]

interactive_requirements = [
    "IPython",
    "ipywidgets",
    "jupyter",
]


setup(
    name=name,
    version=version,
    description=description,
    author=author,
    author_email=author_email,
    url=url,
    license=license,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    tests_require=test_requirements,
    extras_require={
            "interactive": interactive_requirements,
    },
    use_2to3=False,
    zip_safe=True,
)
