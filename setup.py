from setuptools import find_packages, setup

setup(
    name='openloong_retarget',
    version='0.1.0',
	license="BSD-3-Clause",
    packages=find_packages(),
	url="https://github.com/loongOpen/loong_retarget", 
    author='OpenLoong',
    description='retarget tools',
    install_requires=[
    "open3d",
    "numpy==1.23.5",
    "mink",
    "cvxpy",
    "matplotlib",
    "qpsolvers[quadprog]",
    "loop_rate_limiters",
    "smpl_sim@git+https://github.com/ZhengyiLuo/SMPLSim.git@master",
    ]
)
 