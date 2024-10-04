from setuptools import find_packages, setup
package_name = 'rae_oyster'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'image_pipeline', 'face_recognition'],
    zip_safe=True,
    maintainer='Joey',
    maintainer_email='j.g.vanderkaaij@uva.nl',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pearl_node = rae_oyster.pearl_node:main',
            'line_node = rae_oyster.line_node:main'
        ],
    },
)
