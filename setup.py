from setuptools import setup

setup(name='livefeature',
      version='0.1',
      description='Live features for TensorFlow models',
      url='http://github.com/kjchavez/live-feature',
      author='Kevin Chavez',
      author_email='kevin.j.chavez@gmail.com',
      license='MIT',
      packages=['livefeature'],
      install_requirements=["cachetools>=2.0.1"],
      zip_safe=False)
