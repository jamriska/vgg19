@echo off
setlocal ENABLEDELAYEDEXPANSION

for %%V in (14,12,11,10) do if exist "!VS%%V0COMNTOOLS!" call "!VS%%V0COMNTOOLS!..\..\VC\vcvarsall.bat" amd64 && goto compile
echo Unable to detect Visual Studio path!
goto error

:compile
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
cl classify.cpp   /I"." /DNDEBUG /Ox /Oy /Gy /GL /openmp /fp:fast /EHsc /Fe"classify.exe"   /nologo || goto error
cl dumpfeat.cpp   /I"." /DNDEBUG /Ox /Oy /Gy /GL /openmp /fp:fast /EHsc /Fe"dumpfeat.exe"   /nologo || goto error
cl featmosaic.cpp /I"." /DNDEBUG /Ox /Oy /Gy /GL /openmp /fp:fast /EHsc /Fe"featmosaic.exe" /nologo || goto error
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

goto :EOF

:error
echo FAILED
@%COMSPEC% /C exit 1 >nul
