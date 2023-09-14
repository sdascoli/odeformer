from setuptools import setup, find_packages

setup(
    name='odeformer',
    version='0.0.1',
    description="Transformers for symbolic regression of ODEs",
    author="Stéphane d'Ascoli",
    author_email="stephane.dascoli@gmail.com",
    packages="odeformer",
    install_requires=[
        "numexpr==2.8.4",
        "sympy==1.11.1",
        "matplotlib",
        "numpy",
        "pandas",
        "requests",
        "scikit-learn",
        "scipy",
        "seaborn",
        "setproctitle",
        "torch==2.0.0",
        "tqdm",
        "wandb",
        "pysindy"
    ]
)

