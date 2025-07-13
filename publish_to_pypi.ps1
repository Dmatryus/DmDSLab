# Конфигурация
$PackageName = "DmDSLab"
$PyPiRepo = "pypi"
$PyPiToken = $env:PYPI_TOKEN

Write-Host "$PyPiToken"

# Очистка
Remove-Item -Recurse -Force build, dist, *.egg-info -ErrorAction SilentlyContinue

# Получение версии
$Version = python -c "import $PackageName; print($PackageName.__version__)"
Write-Host "Публикация версии: $Version"

# Проверка существующей версии
$Response = Invoke-RestMethod "https://pypi.org/pypi/$PackageName/$Version/json" -ErrorAction SilentlyContinue
if ($null -eq $Response) {
    Write-Host "Версия $Version не найдена на PyPI"
} else {
    Write-Host "ОШИБКА: Версия $Version уже существует!"
    exit 1
}

# Установка зависимостей
python -m pip install --upgrade pip setuptools wheel twine

# Сборка
python setup.py sdist bdist_wheel

# Проверка
twine check dist/*

# Публикация
twine upload --repository $PyPiRepo `
    --username "__token__" `
    --password $PyPiToken `
    dist/*

# Финал
Remove-Item -Recurse -Force build, *.egg-info -ErrorAction SilentlyContinue
Write-Host "Успешно опубликовано: https://pypi.org/project/$PackageName/$Version"
