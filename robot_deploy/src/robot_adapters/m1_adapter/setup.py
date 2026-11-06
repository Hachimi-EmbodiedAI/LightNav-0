from setuptools import find_packages, setup

package_name = "m1_adapter"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="xiaoyang",
    maintainer_email="xiaoyang@lightrobo.com",
    description="Genisom M1 adapter for the generic robot ROS interface.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["m1_adapter = m1_adapter.adapter_node:main"]},
)
