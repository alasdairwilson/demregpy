from setuptools import setup

def readme():
    with open('README.md', 'r') as f:
        return f.read()


setup(
    name='demregpy',
    version='0.6.2',
    description='DEM Regularised Inversion Calculation in Python (Hannah & Kontar 2012)',
    long_description_content_type='text/markdown',
    long_description=readme(),
    url='https://github.com/alasdairwilson/demregpy',
    author='Alasdair Wilson',
    author_email='alasdair.wlsn@gmail.com',
    license='MIT',
    packages=[''],
    include_package_data = True,
    package_data={'': ['tresp/*.dat']},
    install_requires=[
        'numpy',
        'tqdm',
        'threadpoolctl',
    ],
    python_requires='>=3.6',
    clasifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License'
    ],
    zip_safe=False
)