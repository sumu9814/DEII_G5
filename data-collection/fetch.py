import os
import re
import sys
import json
import time
import requests
import datetime
import subprocess


def log(*text, level="INFO", to_file=False):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    padding = max(len(key) for key in LOG_FILES.keys()) + 2
    log_key = level.ljust(padding, " ")
    
    if to_file:
        with open(LOG_FILES[level], 'a') as f:
            print(timestamp, log_key, *text, file=f)
            
    else:
        print(timestamp, log_key, *text)

def paged_get(url, headers, params = {}):
    """
    Make requests to a given endpoint of the Github REST API
    until there are no more pages left.

    Args:
        url (str): _description_
        headers (dict): _description_
        params (dict, optional): Additional parameters for the
            request. They will be added like "ke1y=value1&key2=value2..."
            to the url. Defaults to {}.

    Returns:
        list: each item will be a dictionary with the
        information that was requested of the given entity
        (issues, pull requests, etc.)
        
    """
    
    
    is_error = False
    params["per_page"] = PER_PAGE
    results = []
    page = 1
    
    while True:
        
        params["page"] = page
        params_str = "&".join([f"{key}={value}" for key, value in params.items()])
        
        try:
            r = requests.get(f"{url}?{params_str}", headers=headers)
            
            if r.status_code != 200:
                is_error=True
                log(f"Failed: {url} (status {r.status_code})", level=LOG_ERROR_LEVEL)
                break
            
            data = r.json()
            
        except Exception as e:
            is_error=True
            log(f"Failed: {url} exception {repr(e)}", level=LOG_ERROR_LEVEL)
            break

        if not data:
            break
        
        results.extend(data)
        
        log(f"Fetched page {page} of {url} ({len(data)} items)", level=LOG_INFO_LEVEL)
        
        page += 1
        
        time.sleep(RATE_LIMIT_DELAY)
        
    return results, is_error


# Control variables:
N_REPOS_WITH_ERRORS = 0
MAX_STARS_COUNT = 0


# LOG CONSTANTS
LOG_INFO_LEVEL = "INFO"
LOG_ERROR_LEVEL = "ERROR"
LOG_FILES = {
    LOG_INFO_LEVEL: "info.log",
    LOG_ERROR_LEVEL: "error.log"
}

# Check the required file path to repositories metadata is correct:
if len(sys.argv) != 2:
    print("Usage:", sys.argv[0], "<repositories-data-file-path>")
    log("Could not fetch repositories data. Repositories JSON file path not provided.", level=LOG_ERROR_LEVEL)
    sys.exit(1)

REPOS_JSON = sys.argv[1]
if not os.path.exists(REPOS_JSON):
    log("Provided file", REPOS_JSON, "does not exist.", level=LOG_ERROR_LEVEL)
    sys.exit(1)


GITHUB_API_TOKEN = os.getenv("GH_TOKEN")
if GITHUB_API_TOKEN is None:
    log("GitHub API token environment variable not defined. Set GH_TOKEN environment variable.", level=LOG_ERROR_LEVEL)
    sys.exit(1)

# GITHUB API REQUESTS CONSTANTS
PER_PAGE         = 100
RATE_LIMIT_DELAY = 1.5
HEADERS = {
    "Authorization": f"token {GITHUB_API_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

STARGAZER_HEADERS = {
    **HEADERS,
    "Accept": "application/vnd.github.v3.star+json"
}

with open(REPOS_JSON) as f:
    repos = json.load(f)

for repo in repos:
    owner_repo = repo["fullName"]
    repo_id = re.sub(r'[^0-9a-z]', '', repo['id'].lower())
    
    log(f"Fetching data for {owner_repo}...", level=LOG_INFO_LEVEL)

    owner, name = owner_repo.split("/")
    base_url = f"https://api.github.com/repos/{owner}/{name}"
    

    # FETCH COMMITS:
    commits, is_error = paged_get(f"{base_url}/commits", HEADERS)
    
    if is_error:
        N_REPOS_WITH_ERRORS += 1
        continue
    
    # FETCH STARGAZERS WITH TIMESTAMPS:
    stargazers, is_error = paged_get(f"{base_url}/stargazers", STARGAZER_HEADERS)
    
    if is_error:
        N_REPOS_WITH_ERRORS += 1
        continue
    
    # FETCH ISSUES
    issues, is_error = paged_get(f"{base_url}/issues", HEADERS, {"state": "all"})
    
    if is_error:
        N_REPOS_WITH_ERRORS += 1
        continue
    
            
    # FETCH PULL REQUESTS:
    pulls, is_error = paged_get(f"{base_url}/pulls?state=all", HEADERS)
    
    if is_error:
        N_REPOS_WITH_ERRORS += 1
        continue

    # FETCH CONTRIBUTORS:
    contributors, is_error = paged_get(f"{base_url}/contributors?anon=true", HEADERS)
    
    if is_error:
        N_REPOS_WITH_ERRORS += 1
        continue
    
    
    # Update control variables:
    MAX_STARS_COUNT = max(repo["stargazersCount"], MAX_STARS_COUNT)
    
    # Put it all together in a single json file:
    repo_data = repo.copy()
    repo_data["fetched_commits"] = commits
    repo_data["fetched_stargazers"] = stargazers
    repo_data["fetched_issues"] = issues
    repo_data["fetched_pulls"] = pulls
    repo_data["fetched_contributors"] = contributors
    
    os.makedirs("repos", exist_ok=True)
    with open(f"repos/{repo_id}.json", "w") as f:
        json.dump(repo_data, f, indent=6)


# Maybe a few repositories were not collected due to errors, collect more
if N_REPOS_WITH_ERRORS:
    subprocess.Popen(
        [
            "./fetch.sh",
            str(N_REPOS_WITH_ERRORS), # number of repositories to fetch
            str(MAX_STARS_COUNT + 1), # minimum stars
            str(MAX_STARS_COUNT + 1 + N_REPOS_WITH_ERRORS), # maximum stars
            str(1) # repositories per request
        ],
        start_new_session=True
    )
