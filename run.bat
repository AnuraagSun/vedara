@echo off

echo.
echo =====================================================
echo  VEDARA AR System Launcher (Windows)
echo =====================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Run: python -m venv venv
    echo Then: venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Starting VEDARA...
echo.
python main.py %*

call deactivate

echo.
echo VEDARA session ended.
pause
