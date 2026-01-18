from setuptools import setup, find_packages

setup(
    name="fraud-detection-system",
    version="1.0.0",
    description="Production-ready Real-Time Transaction Fraud Detection System",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(exclude=["tests", "notebooks"]),
    python_requires=">=3.10",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
