from setuptools import find_packages, setup

package_name = 'camera_intel'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='amir',
    maintainer_email='amir@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "intel_pub = camera_intel.intel_pub:main",
            "intel_sub = camera_intel.intel_sub:main",
            'camera_publisher = camera_intel.camera_publisher:main',
            'camera_subscriber = camera_intel.camera_subscriber:main',
            'real_time_test_publisher = camera_intel.real_time_test_publisher:main',
	    'real_time_test_subscriber= camera_intel.real_time_test_subscriber:main',
            'Depth_wcoordinate = camera_intel.Depth_wcoordinate:main'
        ],
    },
)
