from setuptools import setup, find_packages


setup(
    name = 'RMolEncoder',
    version = '1.0.0',
    author = 'Artem Mukanov',
    # url = 'https://github.com/',
    description = 'Chemical reactions and molecules pretrained encoder',
    # license = , 

    zip_safe = False,
    include_package_data = True, 
    packages = find_packages(), 
    python_requires = '>=3.8', 
    install_requires = [
                        'numpy',
                        'chytorch-rxnmap',
                        'chytorch',
                        'lightning', 
                       ], 
)