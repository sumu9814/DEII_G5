#!/bin/bash
# This script requires the library jq and python3 to be installed
# on the machine it is executed and the GH_TOKEN to be defined as
# an environment variable.

# Check the correct amount of parameters is correct:
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <n_repos> <min_stars> <max_stars> <repos_per_step>"
    exit 1
fi

# Define constants
n_repos=$1
min_stars=$2
max_stars=$3
repos_per_step=$4
n_steps=$(( $n_repos / $repos_per_step ))
step_size=$(( ($max_stars - $min_stars) / $n_steps ))

# Create empty file to store basic metadata from GitHub repositories
tmp_filename="tmp-repos-$(date +%s).json"
data_filename="repos-$(date +%s).json"

echo '' > $tmp_filename

for ((i = 0; i < $n_steps; i++)); do
    lower=$(( $min_stars + i * $step_size ))
    upper=$(( $lower + $step_size - 1))

    echo "Fetching repos between stars $lower and $upper"

    gh search repos "stars:$lower..$upper" \
        --limit $repos_per_step \
        --visibility public \
        --json createdAt,defaultBranch,description,forksCount,fullName,hasDownloads,hasIssues,hasPages,hasProjects,hasWiki,homepage,id,isArchived,isDisabled,isFork,isPrivate,language,license,name,openIssuesCount,owner,pushedAt,size,stargazersCount,updatedAt,url,visibility,watchersCount \
        >> $tmp_filename

    # Sleep to prevent API limit
    sleep 2
done

# Merge all the rows of the results file in a single JSON object:
jq -s 'add' $tmp_filename > $data_filename
rm $tmp_filename

# Once the loop is done, call python script to fetch more data of each repository
python3 fetch.py $data_filename