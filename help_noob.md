### \#\# 1. Start Your Own Work (on a New Branch)

Your `master` branch now perfectly mirrors the original project's `master` branch (). To keep it that way, you should **never work directly on your `master` branch again**.

Instead, create a **new branch** for any changes you want to make. This keeps your `master` branch clean and makes future updates from the original repository painless.

**To create and switch to a new branch, run:**

```bash
# The "-b" flag creates the new branch and switches to it
git checkout -b my-notebook-experiments
```

You can name the branch anything you like. Now you are on a safe, separate timeline. You can freely:

  * Modify the Jupyter notebooks.
  * Add your own notes and code.
  * Experiment with the deep learning models.

When you commit your changes, they will be saved to this new branch (`my-notebook-experiments`) without affecting your `master` branch.

-----

### \#\# 2. Contribute Back (Optional)

If you make a change that you think would be a great addition to the original project (like fixing a bug), you can contribute it back.

1.  Push your new branch to your GitHub fork:
    ```bash
    git push origin my-notebook-experiments
    ```
2.  Go to your GitHub repository in your browser. GitHub will automatically show a button to **"Compare & pull request"**.
3.  Click it and write a clear description of your changes to suggest them to François Chollet. This is called opening a **Pull Request**.

-----

### \#\# 3. How to Sync in the Future (The Easy Way) 🚀

The next time Chollet updates his repository, the process to sync your `master` branch will be much simpler because you haven't added any new commits to it.

1.  **Switch back to your `master` branch:**
    ```bash
    git checkout master
    ```
2.  **Pull the latest changes using rebase:**
    ```bash
    git pull --rebase upstream master
    ```
3.  **Update your GitHub fork:**
    ```bash
    git push origin master
    ```

That's it\! Doing this regularly (especially before creating a new feature branch) will prevent the large conflicts you saw before.
To ensure that all new commits go to your new branch (e.g., "my-notebook-experiments") and not to the master branch when you reopen your repo:

### \#\# 4.How to Check and Switch Branches

- When you open your repository, check which branch is currently active by running:
  ```
  git branch
  ```
  The active branch will be marked with an asterisk (*).

- If you are not on your new branch, switch to it with:
  ```
  git checkout my-notebook-experiments
  ```
  Replace "my-notebook-experiments" with your actual branch name.

### \#\# 5. Best Practice

- Always switch to your working branch before making changes or committing.
- You can configure your Git environment or editor (like VS Code) to show the current branch prominently to avoid confusion.
- To confirm where commits will go, use:
  ```
  git status
  ```
  This shows the current branch and staged changes.

### \#\# 6. Reminder

- Your master branch should remain clean, tracking upstream only.
- Work, commit, and experiment on your feature or work branch.

By following this, you will keep new commits safe on your new branch, avoiding accidental commits on master.[1][2]

[1](https://docs.github.com/en/get-started/using-git/using-git-rebase-on-the-command-line)
[2](https://git-scm.com/docs/git-rebase)