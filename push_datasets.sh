#!/bin/bash

# Script to push each commit one by one to Git
# Navigate to project root
cd "/Users/sanadalsharafi/Desktop/Work/digital proj/AI-Project"

echo "Starting to push commits one by one..."
echo "=============================================="

# Get the remote tracking branch (origin/main)
ORIGIN_HEAD=$(git rev-parse origin/main)
LOCAL_HEAD=$(git rev-parse HEAD)

# Get list of commits to push (from origin/main to HEAD, in order from oldest to newest)
COMMITS=$(git rev-list origin/main..HEAD | awk '{line[NR]=$0} END {for(i=NR;i>=1;i--) print line[i]}')

if [ -z "$COMMITS" ]; then
    echo "No commits to push!"
    exit 0
fi

COMMIT_COUNT=$(echo "$COMMITS" | wc -l)
CURRENT=1

echo "Found $COMMIT_COUNT commit(s) to push"
echo ""

# Push each commit one by one
for commit in $COMMITS; do
    COMMIT_MSG=$(git log -1 --pretty=%B "$commit")
    COMMIT_SHORT=$(git rev-parse --short "$commit")
    
    echo "[$CURRENT/$COMMIT_COUNT] Pushing commit: $COMMIT_SHORT - $COMMIT_MSG"
    echo "---"
    
    # Checkout this commit temporarily
    git push origin "$commit:main" --force-with-lease || {
        echo "⚠ Push failed for $COMMIT_SHORT"
        exit 1
    }
    
    echo "✓ Successfully pushed commit $COMMIT_SHORT"
    echo ""
    
    ((CURRENT++))
done

# Reset to main branch
git checkout main
git pull origin main

echo "=============================================="
echo "All $COMMIT_COUNT commits have been pushed successfully!"
