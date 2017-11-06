from setuptools import setup
from setuptools import Extension

about = {}
with open('./pyml/__about__.py', 'r') as f:
    exec(f.read(), about)

linear_algebra_module = Extension('linearAlgebraModule',
                                  sources=['./pyml/maths/src/linearalgebramodule.c'])

gradient_descent_module = Extension('gradientDescentModule',
                                    sources=['./pyml/maths/src/gradientdescentmodule.cpp'],
                                    extra_compile_args=['-std=c++11'])

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
              'pyml.preprocessing',
              'pyml.linear_models',
              'pyml.utils'],
    url='https://github.com/gf712/PyML',
    license=about['__license__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    description='Machine learning with Python and C/C++',
    test_suite="tests",
    ext_modules=[linear_algebra_module, gradient_descent_module]
)
