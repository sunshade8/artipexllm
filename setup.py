from setuptools import setup, find_packages

setup(
    name="adaptive-rag-ko",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langchain-openai>=0.0.1",
        "langgraph>=0.0.15",
        "pydantic>=2.0.0",
        "nbformat",
        "openai",
        "chromadb",
        "tiktoken",
        "tavily-python",
        "python-dotenv",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Korean Adaptive RAG system with multi-turn dialogue support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/adaptive-rag-ko",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "adaptive_rag": ["data/sample/*.pdf", "data/sample/*.txt"],
    },
)
