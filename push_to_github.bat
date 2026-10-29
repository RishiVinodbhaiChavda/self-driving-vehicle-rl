@echo off
REM Quick GitHub Push Script for Windows
REM This script will help you push the project to GitHub

echo ============================================================
echo SELF-DRIVING VEHICLE RL - GITHUB PUSH SCRIPT
echo ============================================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed!
    echo Please install Git from: https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

echo Git is installed. Proceeding...
echo.

REM Check if already a git repository
if not exist ".git" (
    echo Initializing Git repository...
    git init
    echo.
)

REM Get GitHub username
echo Enter your GitHub username:
set /p GITHUB_USER=Username: 

REM Get repository name
echo.
echo Enter repository name (e.g., self-driving-vehicle-rl):
set /p REPO_NAME=Repository name: 

echo.
echo ============================================================
echo CONFIGURATION
echo ============================================================
echo GitHub Username: %GITHUB_USER%
echo Repository Name: %REPO_NAME%
echo Repository URL: https://github.com/%GITHUB_USER%/%REPO_NAME%.git
echo.
echo Is this correct? (Press any key to continue, Ctrl+C to cancel)
pause >nul

REM Add remote (remove if exists)
git remote remove origin >nul 2>&1
echo Adding remote repository...
git remote add origin https://github.com/%GITHUB_USER%/%REPO_NAME%.git
echo.

REM Stage all files
echo Staging files...
git add .
echo.

REM Show status
echo Files to be committed:
git status --short
echo.

REM Get commit message
echo Enter commit message (or press Enter for default):
set /p COMMIT_MSG=Message: 

if "%COMMIT_MSG%"=="" (
    set COMMIT_MSG=Initial commit: Complete self-driving vehicle RL system with lane-following, racing, and track color following
)

REM Commit
echo.
echo Committing files...
git commit -m "%COMMIT_MSG%"
echo.

REM Push
echo Pushing to GitHub...
git branch -M main
git push -u origin main

if errorlevel 1 (
    echo.
    echo ============================================================
    echo PUSH FAILED!
    echo ============================================================
    echo.
    echo Possible reasons:
    echo 1. Repository doesn't exist on GitHub
    echo    - Create it at: https://github.com/new
    echo.
    echo 2. Authentication failed
    echo    - You may need to use a Personal Access Token
    echo    - Create one at: https://github.com/settings/tokens
    echo.
    echo 3. Remote already has commits
    echo    - Try: git pull origin main --allow-unrelated-histories
    echo    - Then: git push -u origin main
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS! Project pushed to GitHub!
echo ============================================================
echo.
echo Your repository is now live at:
echo https://github.com/%GITHUB_USER%/%REPO_NAME%
echo.
echo Next steps:
echo 1. Visit your repository on GitHub
echo 2. Add topics: machine-learning, reinforcement-learning, pytorch
echo 3. Create a release for trained models (if any)
echo 4. Share your project!
echo.
echo ============================================================
pause
