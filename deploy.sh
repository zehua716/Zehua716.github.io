# !/bin/bash

echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"

# Build the project.
hugo -F --cleanDestinationDir # if using a theme, replace with `hugo -t <YOURTHEME>`

# Add changes to git.
git add .

# Commit changes.
msg="update site `date`"
if [ $# -eq 1 ]
  then msg="$1"
fi
git commit -m "$msg"

# Push source and build repos.
git push origin main
