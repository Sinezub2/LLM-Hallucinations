# Hallucination Resistance in Large Language Models

## Problem statement


The example solution could be seen in files solution.py. The environment that it will use is provided in Dockerfile. You can customize Dockerfile to your needs.

## Submitting your solution
1. Run ./init_dvc.sh. This will allow you to properly initiate dvc (data version control) that we will use to store large files (like model weigths)
2. To submit your solution use following commands:
```bash
dvc add <model checkpoint path> # important: checkpoint should be in the same directory as other project files
dvc push  # upload model weights to S3
git add . # commit the rest of files
git commit -m "Solution"
git push
```
Then click use the button to submit your task in contest.yandex.ru
Remember, that there is a limit of to submissions per day

