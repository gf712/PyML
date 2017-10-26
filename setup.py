from setuptools import setup

about = {}
with open('./pyml/__about__.py', 'r') as f:
    exec(f.read(), about)

setup(
    name='PyML',
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Programming Language :: Python :: 3 :: Only'],
    keywords='machine learning',
    version=about['__version__'],
    package_dir={'pyml': 'pyml'},
    packages=['pyml',
              'pyml.maths',
              'pyml.metrics',
              'pyml.nearest_neighbours',
              'pyml.datasets',
              'pyml.preprocessing'],
    url='https://github.com/gf712/PyML',
    license=about['__license__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    description='Pure python machine learning',
    test_suite="tests"
)
