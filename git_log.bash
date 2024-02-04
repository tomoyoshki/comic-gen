#!/bin/bash

# Ensure the script exits if any command fails
set -e

# File to store the git log output
mkdir -p logs

LOG_FILE="logs/git_commits.log"

# Clear the log file in case it already exists
> "$LOG_FILE"

# Fetch all branches
git fetch --all

# Iterate over each branch
for branch in $(git branch -r | grep -v HEAD); do
    # Trim "origin/" from the branch name if needed
    clean_branch=${branch#origin/}

    git checkout "$clean_branch" > /dev/null 2>&1

    git log --pretty=format:'%h,%an,%ar,%s' >> "$LOG_FILE"
    echo "\n" >> "$LOG_FILE"
done

# Input and output files
input_file="./logs/git_commits.log"
output_file="./logs/git_commits_pruned.log"

# Check if input file exists
if [ ! -f "$input_file" ]; then
    echo "Input file does not exist."
    exit 1
fi

# Create or clear the output file
> "$output_file"

# Use awk to filter out duplicate lines based on the first field (commit hash)
awk '!seen[$1]++' "$input_file" > "$output_file"

# Define the input log file and output file
INPUT_FILE="./logs/git_commits_pruned.log"
OUTPUT_FILE="./logs/git_contribution_stats.log"

# Initialize the output file
> "$OUTPUT_FILE"

# Read each line from the input file
while IFS=, read -r commit_hash author date message; do
    # Skip lines that are not commit hashes or are merge commits
    if [[ "$message" == "Merge"* ]]; then
        continue
    fi

    # Use git show to get the number of lines added and removed for this commit
    # --numstat: Show number of lines added and removed
    # --format="%aN": Show the author's name
    echo "Author" >> "$OUTPUT_FILE"
    git show --numstat --format="%aN" "$commit_hash" >> "$OUTPUT_FILE"

done < "$INPUT_FILE"


# echo "Processing complete. Output written to $output_file"

LOG_FILE="./logs/git_contribution_stats.log"

# Define associative arrays for contributors and their aliases
declare -A contributors=(
    ["tomoyoshi"]="tommy|tomoyoshi kimura|tommy kimura"
    ["tianchen"]="tianchen|tianchen wang|tim wang|tim"
    ["weiyang"]="weiyang|weiyang wang|will"
    ["ruipeng"]="ruipeng han|ruipeng|ruipeng2"
    ["kaiyuan"]="kaiyuan|kaiyuan luo|kyluo"
)

# Initialize arrays to store added and deleted lines count
declare -A added_lines
declare -A deleted_lines

# Initialize a variable to keep track of the current author
current_author=""

while IFS= read -r line; do
    if [[ "$line" == "Author" ]]; then
        read -r author_name
        author_name=$(echo "$author_name" | tr '[:upper:]' '[:lower:]')
        current_author=""

        # Match the author name with known aliases
        for contributor in "${!contributors[@]}"; do
            regex="${contributors[$contributor]}"
            if [[ $author_name =~ $regex ]]; then
                current_author=$contributor
                break
            fi
        done

        if [[ -z "$current_author" ]]; then
            continue
        fi

        if [[ -z "${added_lines[$current_author]}" ]]; then
            added_lines[$current_author]=0
            deleted_lines[$current_author]=0
        fi

    elif [[ -n "$current_author" && "$line" =~ ^[0-9]+ ]]; then
        # Split lines
        IFS=$'\t' read -r added deleted filename<<< "$line"

        added_lines["$current_author"]=$((added_lines["$current_author"] + added))
        deleted_lines["$current_author"]=$((deleted_lines["$current_author"] + deleted))

    fi

done < "$LOG_FILE"

CONTRIBUTION_FILE="./logs/git_contribution_line_count.log"
> "$CONTRIBUTION_FILE"
# Print the results
for contributor in "${!contributors[@]}"; do
    echo "$contributor: ${added_lines[$contributor]} line(s) added, ${deleted_lines[$contributor]} line(s) deleted" >> "$CONTRIBUTION_FILE"
done
