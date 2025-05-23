# GitHub Repository Data Collector

This Dockerized application provides a self-contained environment for collecting detailed data from GitHub repositories. It's designed for flexibility, allowing users to define collection parameters and securely store the output files outside the container.

## Features

- Automated Data Collection: Leverages gh command line tool for initial repository searches and a Python script (fetch.py) for comprehensive data retrieval (commits, stargazers, issues, pull requests, contributors).
- Containerized Environment: Encapsulated in Docker for consistent execution across different environments.
- Secure Credential Handling: Utilizes environment variables (GH_TOKEN) for passing GitHub API tokens at runtime, preventing their inclusion in the Docker image layers.
- Configurable Parameters: Supports dynamic specification of repository filtering criteria (e.g., number of repositories, stargazer ranges).
- Persistent Storage: Mappable volume for saving collected JSON data directly to the host machine.
- Error Handling & Retries: Includes logic to track collection errors and initiate a new process to recover incomplete data.

## Prerequisites
- Docker Desktop (or Docker Engine) installed and running on your system.
- A GitHub Personal Access Token with public_repo scope (or appropriate scopes for private repositories if applicable).

## Getting Started

Follow these steps to build and run the Docker application.

### 1. Project Setup

Ensure you have the following files in your project directory:

- Dockerfile
- fetch.sh
- fetch.py

### 2. Build the Docker Image

Navigate to your project directory in the terminal and build the Docker image:

```
docker build -t github-data-collector .
```

### 3. Run the Docker Container

You must provide your GitHub API token as an environment variable and specify a host directory for storing the output. You can also customize the data collection parameters.

#### Required Parameters:

- **GH_TOKEN**: Your GitHub Personal Access Token. This is passed via the -e flag.
- **/host/path/to/output**:/app/repos: A Docker volume mount (-v flag) that maps a directory on your host machine to the /app/repos directory inside the container, where the JSON files will be saved. Replace **/host/path/to/output** with an absolute path to a directory on your machine (e.g., /home/ubuntu/github_data).

#### Optional Collection Parameters (passed directly to docker run):

These parameters correspond to the arguments of fetch.sh:

- **n_repos**: Number of repositories to fetch (defaults to 1000).

- **min_stars**: Minimum number of stargazers (defaults to 50).

- **max_stars**: Maximum number of stargazers (defaults to 2000).

- **repos_per_step**: Repositories to fetch at each step (defaults to 2).

Example Run Command:

```
docker run \
  -e GH_TOKEN="YOUR_GITHUB_API_TOKEN" \
  -v /home/ubuntu/github_repo_data:/app/repos \
  github-data-collector 500 100 5000 5
```

Replace **YOUR_GITHUB_API_TOKEN** with your actual token and **/home/ubuntu/github_repo_data** with your desired output directory on the host.

## How it Works

1. **Initialization**: The container starts, and the ENTRYPOINT executes **fetch.sh**.
2. **Repository Discovery**: **fetch.sh** uses the gh command line tool (authenticated via GH_TOKEN environment variable) to search for repositories based on the provided star range and quantity parameters. It aggregates basic metadata into a temporary JSON file.
3. **Detailed Data Fetching**: **fetch.sh** then calls fetch.py, passing the path to the temporary JSON file argument.
4. **API Calls**: **fetch.py** reads the GH_TOKEN environment variable for authentication. For each repository in the basic metadata, it makes further API calls to retrieve detailed information such as commits, stargazers with timestamps, issues, pull requests, and contributors.
5. **Data Storage**: The comprehensive data for each repository is then saved as a separate JSON file within the /app/repos directory inside the container, which is mounted to your specified host directory.
6. **Error Handling and Re-attempts**: If API errors occur, **fetch.py** tracks the number of incomplete repositories and, if necessary, initiates a new **fetch.sh** process to attempt to collect additional data for the failed count, adjusting star ranges to find new repositories.