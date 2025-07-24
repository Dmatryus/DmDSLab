# DmDSLab PyPI Publishing Script
# Версия: 2.0
# Автор: DmDSLab Team

param(
    [switch]$TestRepo,
    [switch]$SkipVersionCheck,
    [switch]$DryRun
)

# Конфигурация
$PackageName = "DmDSLab"
$PyPiRepo = if ($TestRepo) { "testpypi" } else { "pypi" }
$PyPiToken = $env:PYPI_TOKEN
$TestPyPiToken = $env:TEST_PYPI_TOKEN

Write-Host "DmDSLab PyPI Publishing Script" -ForegroundColor Green
Write-Host "===============================" -ForegroundColor Green

# Проверка токена
$TokenToUse = if ($TestRepo) { $TestPyPiToken } else { $PyPiToken }
if (-not $TokenToUse) {
    $TokenVar = if ($TestRepo) { "TEST_PYPI_TOKEN" } else { "PYPI_TOKEN" }
    Write-Host "ОШИБКА: Переменная окружения $TokenVar не установлена!" -ForegroundColor Red
    Write-Host "Установите токен: `$env:$TokenVar = 'your-token-here'" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Токен найден" -ForegroundColor Green
Write-Host "✓ Репозиторий: $PyPiRepo" -ForegroundColor Green

# Очистка предыдущих сборок
Write-Host "`nОчистка предыдущих сборок..." -ForegroundColor Yellow
Remove-Item -Recurse -Force build, dist, *.egg-info -ErrorAction SilentlyContinue

# Получение версии из пакета
Write-Host "Получение версии пакета..." -ForegroundColor Yellow
try {
    $Version = python -c "import $PackageName; print($PackageName.__version__)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось импортировать пакет"
    }
    Write-Host "✓ Версия для публикации: $Version" -ForegroundColor Green
} catch {
    Write-Host "ОШИБКА: Не удалось получить версию пакета!" -ForegroundColor Red
    Write-Host "Убедитесь, что пакет установлен: pip install -e ." -ForegroundColor Yellow
    exit 1
}

# Проверка существующей версии (если не пропускаем)
if (-not $SkipVersionCheck) {
    Write-Host "`nПроверка существования версии на PyPI..." -ForegroundColor Yellow
    try {
        $ApiUrl = if ($TestRepo) { 
            "https://test.pypi.org/pypi/$PackageName/$Version/json" 
        } else { 
            "https://pypi.org/pypi/$PackageName/$Version/json" 
        }
        
        $Response = Invoke-RestMethod $ApiUrl -ErrorAction Stop
        Write-Host "ОШИБКА: Версия $Version уже существует на $PyPiRepo!" -ForegroundColor Red
        Write-Host "Обновите версию в setup.py и __init__.py" -ForegroundColor Yellow
        exit 1
    } catch {
        if ($_.Exception.Response.StatusCode -eq 404) {
            Write-Host "✓ Версия $Version не найдена на $PyPiRepo - можно публиковать" -ForegroundColor Green
        } else {
            Write-Host "Предупреждение: Не удалось проверить существование версии: $($_.Exception.Message)" -ForegroundColor Yellow
            Write-Host "Продолжаем публикацию..." -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "⚠ Пропускаем проверку версии по запросу" -ForegroundColor Yellow
}

if ($DryRun) {
    Write-Host "`n🔍 РЕЖИМ DRY RUN - публикация не будет выполнена" -ForegroundColor Cyan
}

# Обновление инструментов сборки
Write-Host "`nОбновление инструментов сборки..." -ForegroundColor Yellow
python -m pip install --upgrade pip build twine
if ($LASTEXITCODE -ne 0) {
    Write-Host "ОШИБКА: Не удалось обновить инструменты сборки!" -ForegroundColor Red
    exit 1
}

# Сборка пакета (современный способ)
Write-Host "`nСборка пакета..." -ForegroundColor Yellow
python -m build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ОШИБКА: Сборка пакета не удалась!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Пакет собран" -ForegroundColor Green

# Проверка пакета
Write-Host "`nПроверка пакета..." -ForegroundColor Yellow
twine check dist/*
if ($LASTEXITCODE -ne 0) {
    Write-Host "ОШИБКА: Проверка пакета не прошла!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Пакет прошел проверку" -ForegroundColor Green

# Публикация
if (-not $DryRun) {
    Write-Host "`nПубликация на $PyPiRepo..." -ForegroundColor Yellow
    
    $UploadArgs = @(
        "upload"
        "--repository", $PyPiRepo
        "--username", "__token__"
        "--password", $TokenToUse
        "dist/*"
    )
    
    & twine @UploadArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n🎉 УСПЕШНО ОПУБЛИКОВАНО!" -ForegroundColor Green
        $ProjectUrl = if ($TestRepo) { 
            "https://test.pypi.org/project/$PackageName/$Version" 
        } else { 
            "https://pypi.org/project/$PackageName/$Version" 
        }
        Write-Host "📦 Ссылка: $ProjectUrl" -ForegroundColor Green
        
        if (-not $TestRepo) {
            Write-Host "`n📋 Для установки:" -ForegroundColor Cyan
            Write-Host "pip install $PackageName==$Version" -ForegroundColor White
        }
    } else {
        Write-Host "ОШИБКА: Публикация не удалась!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n✓ DRY RUN завершен - все проверки пройдены" -ForegroundColor Cyan
    Write-Host "Для реальной публикации запустите без флага -DryRun" -ForegroundColor Yellow
}

# Очистка временных файлов
Write-Host "`nОчистка временных файлов..." -ForegroundColor Yellow
Remove-Item -Recurse -Force build, *.egg-info -ErrorAction SilentlyContinue
Write-Host "✓ Очистка завершена" -ForegroundColor Green

Write-Host "`n✨ Скрипт завершен успешно!" -ForegroundColor Green