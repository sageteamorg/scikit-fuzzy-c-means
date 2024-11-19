Installation
============

Using pip
---------

To install **FuzzyCMeans**, follow these steps:

1. Create a virtual environment
   Using `venv`, create a virtual environment to isolate dependencies:

   .. code-block:: bash

       python -m venv .venv
       source .venv/bin/activate  # On macOS/Linux
       # OR
       .venv\Scripts\activate     # On Windows

2. Install the package
   With the virtual environment activated, install the package:

   .. code-block:: bash

       pip install scikit-fuzzy-c-means


Manual Installation
-------------------

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/your-repo/fuzzy-c-means
       cd fuzzy-c-means

2. Create a virtual environment
   Set up a virtual environment for the project:

   .. code-block:: bash

       python -m venv .venv
       source .venv/bin/activate  # On macOS/Linux
       # OR
       .venv\Scripts\activate     # On Windows

3. Install dependencies
   Install the required dependencies:

   .. code-block:: bash

       pip install -r requirements.txt


Using Poetry
------------

You can also use `poetry` for dependency management and environment isolation:

1. Install Poetry
   Make sure `poetry` is installed. If not, you can install it using:

   .. code-block:: bash

       pip install poetry

2. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/your-repo/fuzzy-c-means
       cd fuzzy-c-means

3. Install dependencies and create a virtual environment
   Poetry will handle dependency installation and create a virtual environment automatically:

   .. code-block:: bash

       poetry install

4. Activate the environment
   Activate the virtual environment managed by Poetry:

   .. code-block:: bash

       poetry shell
